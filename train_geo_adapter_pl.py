import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*gradient strides do not match bucket view strides.*")

import argparse
import cv2
import imageio
import os
import wandb
import deepspeed
import random
import shutil
import torch.distributed as distributed
import open3d as o3d
import lightning as pl
import math
import numpy as np
import torch

from deepspeed.ops.adam import FusedAdam
from einops import rearrange
from sklearn.decomposition import PCA
from torchvision import transforms
from torch.utils.data import Dataset
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy
from pathlib import Path
from tqdm import tqdm

from gen3r.models import VGGT, AutoencoderKLWan, GeometryAdapter
from gen3r.models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from gen3r.models.vggt.utils.geometry import unproject_depth_map_to_point_map
from gen3r.utils.loss_utils import compute_camera_loss, compute_depth_loss, compute_point_loss, calculate_adaptive_weight, check_and_fix_inf_nan
from gen3r.utils.data_utils import load_extracted_videos, preprocess_extracted_video_with_resize_crop
from gen3r.utils.eval_utils import compute_chamfer_metrics
from gen3r.utils.common_utils import save_videos_grid, colorize_depth_map


class VideoDataset(Dataset):
    def __init__(self, base_path, num_frames=49, frame_interval=1, height=560, width=560, split='train'):
        self.videos = load_extracted_videos(Path(base_path))
        
        self.split = split
        self.frame_interval = frame_interval
        self.height = height
        self.width = width
        self.target_num_frames = num_frames

        self.__frame_transform = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 )]  # only normalize to 0~1
        )

        if split == 'val':
            num_process = distributed.get_world_size() if distributed.is_initialized() else 8
            self.videos = self.videos[:num_process-2] if num_process > 2 else self.videos[:8]
        else:
            print(f"Number of videos after filtering: {len(self.videos)}")


    def __getitem__(self, index):
        video = self.videos[index]
        frame_interval = random.randint(1, self.frame_interval) if (self.frame_interval > 1 and self.split == 'train') else self.frame_interval
        
        frames, _, _, _ = preprocess_extracted_video_with_resize_crop(
            video, self.height, self.width, num_frames=self.target_num_frames, 
            frame_interval=frame_interval,
            fix_start_frame=0 if self.split == 'val' or self.split == 'test' else None
        )  # [F, C, H, W]

        # Current shape of frames: [F, C, H, W]
        frames = self.__frame_transform(frames).clamp(0, 1)
        data = {"frames": frames}
        
        return data
    

    def __len__(self):
        return len(self.videos)


class GeometryAdapterTrainer(pl.LightningModule):
    def __init__(
        self,
        args,
        learning_rate=1e-5,
        recon_loss_weight=1.,
        similarity_loss_weight=1.,
        align_kl_loss_weight=1.,
        diffusers_checkpoint_path=None,
    ):
        super().__init__()
        
        self.args = args
        if diffusers_checkpoint_path is not None:
            additional_kwargs = {
                "latent_channels": 16,
                "hidden_dim": 128,
                "input_dim": 10240,
                "output_dim": 10240,
                "resample_scale": 10,
            }
            self.model = GeometryAdapter.from_pretrained(diffusers_checkpoint_path, additional_kwargs=additional_kwargs)
        else:
            self.model = GeometryAdapter(latent_channels=16, hidden_dim=128, input_dim=10240, output_dim=10240, resample_scale=10)

        self.learning_rate = learning_rate
        self.recon_loss_weight = recon_loss_weight
        self.similarity_loss_weight = similarity_loss_weight
        self.align_kl_loss_weight = align_kl_loss_weight
        
        self.vggt = VGGT.from_pretrained(args.vggt_path)
        self.aggregator = self.vggt.aggregator
        self.depth_head = self.vggt.depth_head
        self.point_head = self.vggt.point_head
        self.camera_head = self.vggt.camera_head
        self.intermediate_layer_idx = self.vggt.depth_head.intermediate_layer_idx

        self.wan_vae = AutoencoderKLWan.from_pretrained(args.wan_vae_path)

        self.aggregator.eval()
        self.depth_head.eval()
        self.point_head.eval()
        self.camera_head.eval()
        self.wan_vae.eval()

        self.save_hyperparameters()

        del self.vggt
        torch.cuda.empty_cache()

        self.val_step_logs = {}

    
    def _create_optimizer_scheduler(self, params):
        if self.args.use_deepspeed:
            optimizer = FusedAdam(
                params,
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=1e-4
            )
        else:
            optimizer = torch.optim.AdamW(
                params,
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=1e-4
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=self.learning_rate * 0.2
        )
        return optimizer, scheduler


    def configure_optimizers(self):
        trainable_modules = [p[1] for p in self.model.named_parameters() if p[1].requires_grad]
        optimizer, scheduler = self._create_optimizer_scheduler(trainable_modules)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "step",
            }
        }

    
    def train_dataloader(self):
        self.train_dataset = VideoDataset(
            self.args.dataset_path,
            num_frames=self.args.video_sample_n_frames,
            frame_interval=self.args.video_sample_stride,
            width=self.args.video_sample_size,
            height=self.args.video_sample_size,
            split='train',
        )

        # Add distributed sampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_dataset, drop_last=False, shuffle=True) if self.args.distributed or self.args.use_deepspeed else None
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2,  # Number of batches loaded in advance by each worker,
        )
        return train_dataloader
    

    def val_dataloader(self):
        self.val_dataset = VideoDataset(
            self.args.dataset_path.replace('train', 'test'),
            num_frames=self.args.video_sample_n_frames,
            frame_interval=self.args.video_sample_stride,
            width=self.args.video_sample_size,
            height=self.args.video_sample_size,
            split='val',
        )

        # Add distributed sampler
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            self.val_dataset, drop_last=False, shuffle=False) if self.args.distributed or self.args.use_deepspeed else None
        val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            sampler=val_sampler,
            shuffle=False,
            batch_size=1,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )
        return val_dataloader

    
    def on_save_checkpoint(self, checkpoint):
        keys_to_remove = []
        for key in checkpoint['state_dict'].keys():
            if not key.startswith('model.'):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del checkpoint['state_dict'][key]        

    
    def get_aggregated_token_list(self, reconstructed_cam_tokens, reconstructed_image_tokens, target_cam_tokens, target_tokens_list, ps_idx, B, F, D):
        # reshape back to 4 * [B, F, N, D]
        reconstructed_cam_tokens = reconstructed_cam_tokens.mean(dim=(2, 3)).unsqueeze(2).repeat(1, 1, ps_idx, 1)  # [B, F, ps_idx, D]
        reconstructed_aggregated_token_list = [
            torch.cat([torch.zeros_like(reconstructed_cam_tokens), reconstructed_image_tokens[:, :, :, :, :D].reshape(B, F, -1, D)], dim=2),
            torch.cat([torch.zeros_like(reconstructed_cam_tokens), reconstructed_image_tokens[:, :, :, :, D:2*D].reshape(B, F, -1, D)], dim=2),
            torch.cat([torch.zeros_like(reconstructed_cam_tokens), reconstructed_image_tokens[:, :, :, :, 2*D:3*D].reshape(B, F, -1, D)], dim=2),
            torch.cat([reconstructed_cam_tokens, reconstructed_image_tokens[:, :, :, :, 3*D:4*D].reshape(B, F, -1, D)], dim=2),
        ]
        gt_cam_tokens = rearrange(target_cam_tokens, 'b c f h w -> b f h w c').mean(dim=(2, 3)).unsqueeze(2).repeat(1, 1, ps_idx, 1)  # [B, F, ps_idx, D]
        gt_aggregated_token_list = [
            torch.cat([torch.zeros_like(gt_cam_tokens), target_tokens_list[0].reshape(B, F, -1, D)], dim=2),
            torch.cat([torch.zeros_like(gt_cam_tokens), target_tokens_list[1].reshape(B, F, -1, D)], dim=2),
            torch.cat([torch.zeros_like(gt_cam_tokens), target_tokens_list[2].reshape(B, F, -1, D)], dim=2),
            torch.cat([gt_cam_tokens, target_tokens_list[3].reshape(B, F, -1, D)], dim=2),
        ]
        return reconstructed_aggregated_token_list, gt_aggregated_token_list


    def on_train_start(self):
        if self.args.resume_from is not None:
            self.trainer.progress_bar_callback.on_train_epoch_start(self.trainer)


    def training_step(self, batch, batch_idx):

        # Data
        frames = batch["frames"].to(self.device)  # [B, F, C, H, W]
        frames = check_and_fix_inf_nan(frames, loss_name="frames", hard_max=None)
        B, F = frames.shape[:2]

        # normalize to [-1, 1]
        wan_frames = rearrange((frames * 2 - 1).clamp(-1, 1), 'b f c h w -> b c f h w')
        
        with torch.no_grad():
            target_tokens_list, ps_idx = self.aggregator(frames)  # 24 * [B, F, N, D], where N = ps_idx+h*w
            wan_posterior = self.wan_vae.encode(wan_frames).latent_dist  # sample size is [B, 16, f, 70, 70]

        h, w = int(math.sqrt(target_tokens_list[0].shape[-2]-ps_idx)), int(math.sqrt(target_tokens_list[0].shape[-2]-ps_idx))
        D = target_tokens_list[0].shape[-1]  # 2048

        # camera tokens, [B, F, 1, D]
        target_cam_tokens = target_tokens_list[-1][:, :, :1]
        # broadcast the camera tokens to [B, F, h, w, D]
        target_cam_tokens = target_cam_tokens.unsqueeze(2).repeat(1, 1, h, w, 1)

        # convert to 4 * [B, F, h, w, D] and ensure contiguous layout
        target_tokens_list = [
            rearrange(target_tokens_list[i][:, :, ps_idx:], 'b f (h w) c -> b f h w c', h=h, w=w) for i in self.intermediate_layer_idx
        ] 
        target_tokens_list.append(target_cam_tokens)  # 5 * [B, F, h, w, D]
        
        target_tokens = rearrange(torch.cat(target_tokens_list, dim=-1), 'b f h w c -> b c f h w')  # [B, 10240, F, 40, 40]
        target_cam_tokens = rearrange(target_cam_tokens, 'b f h w c -> b c f h w')  # [B, 2048, F, 40, 40]

        # Reconstruct the target tokens
        posterior = self.model.encode(target_tokens).latent_dist
        z = posterior.sample()  # [B, 16, f, 70, 70]
        reconstructed_tokens = self.model.decode(z).sample  # [B, 10240, F, 40, 40]

        # split the reconstructed tokens into image tokens and camera tokens
        reconstructed_image_tokens = reconstructed_tokens[:, :4*D]  # [B, 8192, F, 40, 40]
        reconstructed_cam_tokens = reconstructed_tokens[:, 4*D:]  # [B, 2048, F, 40, 40]

        # Compute losses
        # 1. geometry tokens
        image_token_loss = torch.nn.functional.smooth_l1_loss(reconstructed_image_tokens, target_tokens[:, :4*D], beta=0.15).mean()
        camera_token_loss = torch.nn.functional.smooth_l1_loss(
            reconstructed_cam_tokens.mean(dim=(3, 4)), target_cam_tokens.mean(dim=(3, 4)), beta=0.1).mean()

        # 2. camera parameters, depth map and point map
        reconstructed_aggregated_token_list, gt_aggregated_token_list = self.get_aggregated_token_list(
            rearrange(reconstructed_cam_tokens, 'b c f h w -> b f h w c'),  # [B, F, h, w, D]
            rearrange(reconstructed_image_tokens, 'b c f h w -> b f h w c'),  # [B, F, h, w, D]
            target_cam_tokens,  # [B, D, F, h, w]
            target_tokens_list,  # 5 * [B, F, h, w, D]
            ps_idx, B, F, D
        )
        reconstructed_pose_enc = self.camera_head(reconstructed_aggregated_token_list)
        reconstructed_depth_map, _ = self.depth_head(reconstructed_aggregated_token_list, frames, ps_idx)
        reconstructed_point_map, _ = self.point_head(reconstructed_aggregated_token_list, frames, ps_idx)
        
        with torch.no_grad():
            gt_pose_enc = self.camera_head(gt_aggregated_token_list)
            gt_depth_map, _ = self.depth_head(gt_aggregated_token_list, frames, ps_idx)
            gt_point_map, _ = self.point_head(gt_aggregated_token_list, frames, ps_idx)
        
        camera_loss = compute_camera_loss(reconstructed_pose_enc, gt_pose_enc, loss_type="l1")
        depth_loss = compute_depth_loss(reconstructed_depth_map, gt_depth_map, power=False)
        point_loss = compute_point_loss(reconstructed_point_map, gt_point_map, power=False)

        recon_loss = image_token_loss + camera_token_loss + 10 * camera_loss + 2 * depth_loss + point_loss

        # 3. cosine similarity loss
        image_similarity_loss = 1 - torch.nn.functional.cosine_similarity(reconstructed_image_tokens, target_tokens[:, :4*D], dim=1).mean()
        camera_similarity_loss = 1 - torch.nn.functional.cosine_similarity(reconstructed_cam_tokens.mean(dim=(3, 4)), target_cam_tokens.mean(dim=(3, 4)), dim=1).mean()
        similarity_loss = image_similarity_loss + camera_similarity_loss
        
        loss = self.recon_loss_weight * recon_loss + self.similarity_loss_weight * similarity_loss
        
        # 4. align kl loss
        if self.align_kl_loss_weight > 0:
            kl_loss = posterior.kl(wan_posterior).mean()
            # mean_loss = torch.nn.functional.mse_loss(posterior.mean, wan_posterior.mean).mean()
            # std_loss = torch.nn.functional.mse_loss(posterior.logvar, wan_posterior.logvar).mean()
            with torch.no_grad():
                kl_weight = calculate_adaptive_weight(self.recon_loss_weight * recon_loss, kl_loss) * 0.1
                # mean_weight = calculate_adaptive_weight(self.recon_loss_weight * recon_loss, mean_loss)
                # std_weight = calculate_adaptive_weight(self.recon_loss_weight * recon_loss, std_loss)
            # align_loss = kl_weight * kl_loss + mean_weight * mean_loss + std_weight * std_loss
            align_loss = kl_weight * kl_loss
            loss += self.align_kl_loss_weight * align_loss
        else:
            kl_loss = posterior.kl().mean()
            with torch.no_grad():
                kl_weight = calculate_adaptive_weight(self.recon_loss_weight * recon_loss, kl_loss)
            loss = loss + kl_weight * kl_loss

        # mean and std of the posterior
        mean, mean_std, std = posterior.mean.mean(), posterior.mean.std(), posterior.std.mean()
        # mean and std of the wan_posterior
        wan_mean, wan_mean_std, wan_std = wan_posterior.mean.mean(), wan_posterior.mean.std(), wan_posterior.std.mean()

        # Record log
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/image_token_loss", image_token_loss, prog_bar=False, sync_dist=True)
        self.log("train/camera_token_loss", camera_token_loss, prog_bar=False, sync_dist=True)
        self.log("train/camera_loss", camera_loss, prog_bar=False, sync_dist=True)
        self.log("train/depth_loss", depth_loss, prog_bar=False, sync_dist=True)
        self.log("train/point_loss", point_loss, prog_bar=False, sync_dist=True)
        self.log("train/recon_loss", self.recon_loss_weight * recon_loss, prog_bar=False, sync_dist=True)
        if self.align_kl_loss_weight > 0:
            # self.log("train/mean_loss", mean_weight * mean_loss, prog_bar=False, sync_dist=True)
            # self.log("train/std_loss", std_weight * std_loss, prog_bar=False, sync_dist=True)
            self.log("train/align_loss", self.align_kl_loss_weight * align_loss, prog_bar=False, sync_dist=True)
        self.log("train/mean", mean, prog_bar=False, sync_dist=True)
        self.log("train/mean_std", mean_std, prog_bar=False, sync_dist=True)
        self.log("train/std", std, prog_bar=False, sync_dist=True)
        self.log("train/mean_wan", wan_mean, prog_bar=False, sync_dist=True)
        self.log("train/mean_std_wan", wan_mean_std, prog_bar=False, sync_dist=True)
        self.log("train/std_wan", wan_std, prog_bar=False, sync_dist=True)

        loss = check_and_fix_inf_nan(loss, loss_name="loss", hard_max=None)

        return loss

    
    def on_after_backward(self):
        total_norm = 0.0
        for p in self.model.parameters():
            if self.args.use_deepspeed:
                grad_data = deepspeed.utils.safe_get_full_grad(p)
            else:
                grad_data = p.grad
            if grad_data is not None:
                param_norm = grad_data.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log("train/grad_norm", total_norm)
        
        return 


    def on_validation_epoch_start(self):
        if torch.distributed.is_initialized():
            if not self.trainer.is_global_zero:
                return

        # save the model
        try:
            os.makedirs(os.path.join(self.logger.save_dir, 'ckpt_diffusers', f'checkpoint-{self.global_step}'), exist_ok=True)
            # only remain top 3 checkpoint dirs 
            checkpoint_dir = os.path.join(self.logger.save_dir, 'ckpt_diffusers')
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint-')]
            checkpoint_files.sort(key=lambda x: int(x.split('-')[1]))
            if len(checkpoint_files) > 3:
                try:
                    for file in checkpoint_files[:-3]:
                        shutil.rmtree(os.path.join(checkpoint_dir, file))
                except Exception as e:
                    print(f"Error removing checkpoint: {e}")

            self.model.save_pretrained_safetensors(os.path.join(self.logger.save_dir, f'ckpt_diffusers/checkpoint-{self.global_step}'))

        except Exception as e:
            print(f"Error saving checkpoint: {e}")

        if isinstance(self.logger, pl.pytorch.loggers.WandbLogger):
            self.logger.experiment.define_metric("media_step")
            self.logger.experiment.define_metric("media/*", step_metric="media_step")


    def validation_step(self, batch, batch_idx):

        try:
            batch_idx = batch_idx * torch.distributed.get_world_size() + self.trainer.global_rank
            if batch_idx >= len(self.trainer.fit_loop.epoch_loop.val_loop._data_source.instance.val_dataset):
                return torch.tensor(0.)
        except:
            batch_idx = batch_idx
        
        # Data
        frames = batch["frames"].to(self.device)  # [B, F, C, H, W]
        frames = check_and_fix_inf_nan(frames, loss_name="frames", hard_max=None)
        
        B, F = frames.shape[:2]
        assert B == 1

        # normalize to [-1, 1]
        wan_frames = rearrange((frames * 2 - 1).clamp(-1, 1), 'b f c h w -> b c f h w')
        
        with torch.no_grad():
            target_tokens_list, ps_idx = self.aggregator(frames)  # 24 * [B, F, N, D], where N = ps_idx+h*w
            wan_posterior = self.wan_vae.encode(wan_frames).latent_dist  # sample size is [B, 16, f, 70, 70]
            wan_latents = wan_posterior.mode()  # [B, 16, f, 70, 70]

        h, w = int(math.sqrt(target_tokens_list[0].shape[-2]-ps_idx)), int(math.sqrt(target_tokens_list[0].shape[-2]-ps_idx))
        D = target_tokens_list[0].shape[-1]  # 2048

        # seperate camera and register tokens from the last layer, [B, F, 1, D]
        target_cam_tokens = target_tokens_list[-1][:, :, :1]
        # broadcast the camera tokens to [B, F, h, w, D]
        target_cam_tokens = target_cam_tokens.unsqueeze(2).repeat(1, 1, h, w, 1)

        # convert to 4 * [B, F, h, w, D]
        target_tokens_list = [
            rearrange(target_tokens_list[i][:, :, ps_idx:], 'b f (h w) c -> b f h w c', h=h, w=w) for i in self.intermediate_layer_idx]
        target_tokens_list.append(target_cam_tokens)  # 5 * [B, F, h, w, D]
        
        # concat the tokens along the channel dimension
        target_tokens = rearrange(torch.cat(target_tokens_list, dim=-1), 'b f h w c -> b c f h w')  # [B, 10240, F, 40, 40]
        target_cam_tokens = rearrange(target_cam_tokens, 'b f h w c -> b c f h w')  # [B, 2048, F, 40, 40]

        # Reconstruct
        posterior = self.model.encode(target_tokens).latent_dist
        z = posterior.sample()  # [B, 16, f, 70, 70]
        reconstructed_tokens = self.model.decode(z).sample  # [B, 10240, F, 40, 40]

        # split the reconstructed tokens into image tokens and camera tokens
        reconstructed_image_tokens = reconstructed_tokens[:, :4*D]  # [B, 8192, F, 40, 40]
        reconstructed_cam_tokens = reconstructed_tokens[:, 4*D:]  # [B, 2048, F, 40, 40]

        # Compute losses
        # 1. geometry tokens
        image_token_loss = torch.nn.functional.smooth_l1_loss(reconstructed_image_tokens, target_tokens[:, :4*D], beta=0.15).mean()
        camera_token_loss = torch.nn.functional.smooth_l1_loss(
            reconstructed_cam_tokens.mean(dim=(3, 4)), target_cam_tokens.mean(dim=(3, 4)), beta=0.1).mean()

        # 2. camera parameters, depth map and point map
        reconstructed_aggregated_token_list, gt_aggregated_token_list = self.get_aggregated_token_list(
            rearrange(reconstructed_cam_tokens, 'b c f h w -> b f h w c'),  # [B, F, h, w, D]
            rearrange(reconstructed_image_tokens, 'b c f h w -> b f h w c'),  # [B, F, h, w, D]
            target_cam_tokens,  # [B, D, F, h, w]
            target_tokens_list,  # 5 * [B, F, h, w, D]
            ps_idx, B, F, D
        )
        reconstructed_pose_enc = self.camera_head(reconstructed_aggregated_token_list)
        reconstructed_depth_map, _ = self.depth_head(reconstructed_aggregated_token_list, frames, ps_idx)
        reconstructed_point_map, _ = self.point_head(reconstructed_aggregated_token_list, frames, ps_idx)
        
        with torch.no_grad():
            gt_pose_enc = self.camera_head(gt_aggregated_token_list)
            gt_depth_map, _ = self.depth_head(gt_aggregated_token_list, frames, ps_idx)
            gt_point_map, _ = self.point_head(gt_aggregated_token_list, frames, ps_idx)
        
        camera_loss = compute_camera_loss(reconstructed_pose_enc, gt_pose_enc, loss_type="l1")
        depth_loss = compute_depth_loss(reconstructed_depth_map, gt_depth_map, power=False)
        point_loss = compute_point_loss(reconstructed_point_map, gt_point_map, power=False)

        recon_loss = image_token_loss + camera_token_loss + 10 * camera_loss + 2 * depth_loss + point_loss

        # 3. cosine similarity loss
        image_similarity_loss = 1 - torch.nn.functional.cosine_similarity(reconstructed_image_tokens, target_tokens[:, :4*D], dim=1).mean()
        camera_similarity_loss = 1 - torch.nn.functional.cosine_similarity(reconstructed_cam_tokens.mean(dim=(3, 4)), target_cam_tokens.mean(dim=(3, 4)), dim=1).mean()
        similarity_loss = image_similarity_loss + camera_similarity_loss
        
        loss = self.recon_loss_weight * recon_loss + self.similarity_loss_weight * similarity_loss
        
        # 4. align kl loss
        if self.align_kl_loss_weight > 0:
            kl_loss = posterior.kl(wan_posterior).mean()
            # mean_loss = torch.nn.functional.mse_loss(posterior.mean, wan_posterior.mean).mean()
            # std_loss = torch.nn.functional.mse_loss(posterior.logvar, wan_posterior.logvar).mean()
            with torch.no_grad():
                kl_weight = calculate_adaptive_weight(self.recon_loss_weight * recon_loss, kl_loss) * 0.1
                # mean_weight = calculate_adaptive_weight(self.recon_loss_weight * recon_loss, mean_loss)
                # std_weight = calculate_adaptive_weight(self.recon_loss_weight * recon_loss, std_loss)
            # align_loss = kl_weight * kl_loss + mean_weight * mean_loss + std_weight * std_loss
            align_loss = kl_weight * kl_loss
            loss += self.align_kl_loss_weight * align_loss
        else:
            kl_loss = posterior.kl().mean()
            with torch.no_grad():
                kl_weight = calculate_adaptive_weight(self.recon_loss_weight * recon_loss, kl_loss)
            loss = loss + kl_weight * kl_loss

        # mean and std of the posterior
        mean, mean_std, std = posterior.mean.mean(), posterior.mean.std(), posterior.std.mean()
        # mean and std of the wan_posterior
        wan_mean, wan_mean_std, wan_std = wan_posterior.mean.mean(), wan_posterior.mean.std(), wan_posterior.std.mean()

        loss = check_and_fix_inf_nan(loss, loss_name="loss", hard_max=None)

        # Record log
        self.val_step_logs = {
            'val/loss': loss.item(),
            'val/image_token_loss': image_token_loss.item(),
            'val/camera_token_loss': camera_token_loss.item(),
            'val/camera_loss': camera_loss.item(),
            'val/depth_loss': depth_loss.item(),
            'val/point_loss': point_loss.item(),
            'val/recon_loss': self.recon_loss_weight * recon_loss.item(),
            'val/image_similarity_loss': image_similarity_loss.item(),
            'val/camera_similarity_loss': camera_similarity_loss.item(),
            'val/similarity_loss': self.similarity_loss_weight * similarity_loss.item(),
            'val/mean_wan': wan_mean.item(),
            'val/mean_std_wan': wan_mean_std.item(),
            'val/std_wan': wan_std.item(),
        }
        self.val_step_logs['val/mean'] = mean.item()
        self.val_step_logs['val/mean_std'] = mean_std.item()
        self.val_step_logs['val/std'] = std.item()
        if self.align_kl_loss_weight > 0:
            # self.val_step_logs['val/mean_loss'] = mean_weight * mean_loss.item()
            # self.val_step_logs['val/std_loss'] = std_weight * std_loss.item()
            self.val_step_logs['val/align_kl_loss'] = self.align_kl_loss_weight * align_loss.item()
        
        # Visulizations
        reconstructed_image_tokens = rearrange(reconstructed_image_tokens, 'b c f h w -> b f h w c')  # [B, F, 40, 40, 8192]
        reconstructed_cam_tokens = rearrange(reconstructed_cam_tokens, 'b c f h w -> b f h w c')  # [B, F, 40, 40, 2048]
        
        # PCA of reconstructed tokens, gt tokens, adapted tokens and wan latents
        self.visualize_features(batch_idx, frames.float(), reconstructed_image_tokens, prefix='recon')
        self.visualize_features(batch_idx, frames.float(), rearrange(target_tokens[:, :4*D], 'b c f h w -> b f h w c'), prefix='gt')
        self.visualize_features(batch_idx, frames.float(), rearrange(z, 'b c f h w -> b f h w c'), prefix='ada')
        self.visualize_features(batch_idx, frames.float(), rearrange(wan_latents, 'b c f h w -> b f h w c'), prefix='wan')

        # geometry
        reconstructed_aggregated_token_list, gt_aggregated_token_list = self.get_aggregated_token_list(
            reconstructed_cam_tokens, reconstructed_image_tokens, target_cam_tokens, target_tokens_list, ps_idx, B, F, D
        )
        reconstructed_point_map, reconstructed_point_map_by_unprojection = self.get_geometry(
            frames.float(), reconstructed_aggregated_token_list, batch_idx, prefix='recon')
        gt_point_map, gt_point_map_by_unprojection = self.get_geometry(
            frames.float(), gt_aggregated_token_list, batch_idx, prefix='gt')

        # compute accuracy, completeness and chamfer distance
        point_map_accuracy, point_map_completeness, point_map_chamfer_distance, point_map_relative_percent, _ = \
            compute_chamfer_metrics(reconstructed_point_map, gt_point_map)
        point_map_by_unprojection_accuracy, point_map_by_unprojection_completeness, point_map_by_unprojection_chamfer_distance, point_map_by_unprojection_relative_percent, _ = \
            compute_chamfer_metrics(reconstructed_point_map_by_unprojection, gt_point_map_by_unprojection)

        self.val_step_logs['val/point_map_accuracy'] = point_map_accuracy.item()
        self.val_step_logs['val/point_map_completeness'] = point_map_completeness.item()
        self.val_step_logs['val/point_map_chamfer_distance'] = point_map_chamfer_distance.item()
        self.val_step_logs['val/point_map_relative_percent'] = point_map_relative_percent.item()
        self.val_step_logs['val/point_map_by_unprojection_accuracy'] = point_map_by_unprojection_accuracy.item()
        self.val_step_logs['val/point_map_by_unprojection_completeness'] = point_map_by_unprojection_completeness.item()
        self.val_step_logs['val/point_map_by_unprojection_chamfer_distance'] = point_map_by_unprojection_chamfer_distance.item()
        self.val_step_logs['val/point_map_by_unprojection_relative_percent'] = point_map_by_unprojection_relative_percent.item()

        return loss
    

    # log metrics and save the model in the diffusers style
    def on_validation_epoch_end(self):

        if torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, self.val_step_logs)
            torch.distributed.barrier()
            if not self.trainer.is_global_zero:
                # Clear the outputs for next epoch
                self.val_step_logs = {}
                return
            # merge in rank 0
            self.val_step_logs = {k: [sublist[k] for sublist in gathered] for k in self.val_step_logs}

        for key in self.val_step_logs:
            # average the metrics
            self.val_step_logs[key] = sum(self.val_step_logs[key]) / len(self.val_step_logs[key])
            # log the metrics
            self.logger.log_metrics({key: self.val_step_logs[key]}, step=self.global_step)
        self.val_step_logs = {}


    def visualize_features(self, batch_idx, frames, features, prefix='recon'):
        B, F, h, w, C = features.shape
        assert B == 1

        # [B, F, 40, 40, 8192] for recon and gt, [B, f, 70, 70, 16] for ada and wan
        pca = PCA(n_components=3)
        for i in range(features.shape[1]):
            if prefix == 'recon' or prefix == 'gt':
                feature = features[:, i].float()  # [B, 40, 40, 4D=8192]
                # turn into 4 chunks 
                feature_chunks = torch.chunk(feature, 4, dim=-1)
                feature_pca_list = []
                for j in range(4):
                    feature_chunk = feature_chunks[j]  # [B, 40, 40, 2048]
                    feature_pca = pca.fit_transform(feature_chunk.reshape(-1, feature_chunk.shape[-1]).cpu().numpy())
                    # normalize to [0, 1]
                    min_val = feature_pca.min(axis=(0, 1), keepdims=True)
                    max_val = feature_pca.max(axis=(0, 1), keepdims=True)
                    feature_pca = (feature_pca - min_val) / (max_val - min_val)  # [B * h * w, 3]
                    feature_pca = feature_pca.reshape(h, w, 3)  # [h, w, 3]
                    feature_pca_list.append(feature_pca)

                feature_pca = np.concatenate([
                    np.concatenate(
                        [cv2.resize(feature_pca_list[0], (frames.shape[4], frames.shape[3]), interpolation=cv2.INTER_NEAREST),
                         cv2.resize(feature_pca_list[1], (frames.shape[4], frames.shape[3]), interpolation=cv2.INTER_NEAREST)], axis=1),
                    np.concatenate(
                        [cv2.resize(feature_pca_list[2], (frames.shape[4], frames.shape[3]), interpolation=cv2.INTER_NEAREST),
                         cv2.resize(feature_pca_list[3], (frames.shape[4], frames.shape[3]), interpolation=cv2.INTER_NEAREST)], axis=1)
                ], axis=0)  # [80, 80, 3]

                save_path = os.path.join(
                    self.logger.save_dir, f'validations/step_{self.global_step}/feature_pca/{prefix}', f"rank_{self.trainer.global_rank}", f"batch_{batch_idx}", f"{prefix}_feature_{i}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                imageio.imwrite(save_path, (feature_pca * 255).astype(np.uint8))

                input_frame = rearrange(frames[0, i], 'c h w -> h w c')
                save_path = os.path.join(
                    self.logger.save_dir, f'validations/step_{self.global_step}/feature_pca/{prefix}', f"rank_{self.trainer.global_rank}", f"batch_{batch_idx}", f"{prefix}_input_{i}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                imageio.imwrite(save_path, (input_frame.cpu().numpy() * 255).astype(np.uint8))
            
            elif prefix == 'ada' or prefix == 'wan':
                feature = features[:, i].float()  # [B, 70, 70, 16]
                feature_pca = pca.fit_transform(feature.reshape(-1, feature.shape[-1]).cpu().numpy())
                # normalize to [0, 1]
                min_val = feature_pca.min(axis=(0, 1), keepdims=True)
                max_val = feature_pca.max(axis=(0, 1), keepdims=True)
                feature_pca = (feature_pca - min_val) / (max_val - min_val)  # [B * h * w, 3]
                feature_pca = feature_pca.reshape(h, w, 3)  # [h, w, 3]

                save_path = os.path.join(
                    self.logger.save_dir, f'validations/step_{self.global_step}/feature_pca/{prefix}', f"rank_{self.trainer.global_rank}", f"batch_{batch_idx}", f"{prefix}_feature_{i}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # resize to the same size as the input image based on nearest neighbor
                feature_pca = cv2.resize(feature_pca, (frames.shape[4], frames.shape[3]), interpolation=cv2.INTER_NEAREST)
                imageio.imwrite(save_path, (feature_pca * 255).astype(np.uint8))

        if prefix == 'ada' or prefix == 'wan':
            for i in range(frames.shape[1]):
                input_frame = rearrange(frames[0, i], 'c h w -> h w c')
                save_path = os.path.join(
                    self.logger.save_dir, f'validations/step_{self.global_step}/feature_pca/{prefix}', f"rank_{self.trainer.global_rank}", f"batch_{batch_idx}", f"{prefix}_input_{i}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                imageio.imwrite(save_path, (input_frame.cpu().numpy() * 255).astype(np.uint8))


    def get_geometry(self, frames, aggregated_token_list, batch_idx, prefix='recon'):
        with torch.no_grad():
            pose_enc = self.camera_head(aggregated_token_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, frames.shape[-2:])

            depth_map, _ = self.depth_head(aggregated_token_list, frames, 5)
            point_map, _ = self.point_head(aggregated_token_list, frames, 5)  # [B, F, H, W, 3]
            point_map_by_unprojection = torch.from_numpy(
                unproject_depth_map_to_point_map(depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0))
            ).to(point_map.device)

        # save rgb and depth maps
        save_path = os.path.join(
            self.logger.save_dir, f'validations/step_{self.global_step}/rgb_depth_map/{prefix}', f"rank_{self.trainer.global_rank}_{prefix}_rgb_depth_map_{batch_idx}.mp4")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        colored_depth_map = colorize_depth_map(depth_map.squeeze())[None, ...]  # [B, F, H, W, 3]
        combined = torch.cat([frames.permute(0,1,3,4,2).float(), colored_depth_map], dim=-2).cpu()  # [B, F, H, 4W, 3]
        save_videos_grid(rearrange(combined, "b f h w c -> b c f h w"), save_path, rescale=False)
        
        if isinstance(self.logger, pl.pytorch.loggers.TensorBoardLogger):
            self.logger.experiment.add_video(
                f"media/rank_{self.trainer.global_rank}_{prefix}_rgb_depth_map_{batch_idx}",
                rearrange(combined, "b f h w c -> b f c h w"),
                global_step=self.global_step,
                fps=4
            )
        elif isinstance(self.logger, pl.pytorch.loggers.WandbLogger):
            self.logger.experiment.log(
                {
                    f"media/rank_{self.trainer.global_rank}_{prefix}_rgb_depth_map_{batch_idx}": wandb.Video(save_path), 
                    "media_step": self.global_step
                },
            )

        # save the point map
        save_path = os.path.join(
            self.logger.save_dir, f'validations/step_{self.global_step}/point_map/{prefix}', f"rank_{self.trainer.global_rank}_{prefix}_point_map_{batch_idx}.ply")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # downsample the point map using open3d
        point_map_flat = point_map.view(-1, 3).contiguous().float()
        color = torch.clip(rearrange(frames, 'b f c h w -> (b f h w) c').float(), 0.0, 1.0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_map_flat.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(color.cpu().numpy())
        pcd_downsampled = pcd.voxel_down_sample_and_trace(voxel_size=0.004, min_bound=pcd.get_min_bound(), max_bound=pcd.get_max_bound())[0]
        o3d.io.write_point_cloud(save_path, pcd_downsampled)

        # save the point map by unprojection
        save_path = os.path.join(
            self.logger.save_dir, f'validations/step_{self.global_step}/point_map_by_unprojection/{prefix}', f"rank_{self.trainer.global_rank}_{prefix}_point_map_unprojection_{batch_idx}.ply")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # downsample the point map using open3d
        point_map_by_unprojection_flat = point_map_by_unprojection.view(-1, 3).contiguous().float()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_map_by_unprojection_flat.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(color.cpu().numpy())
        pcd_downsampled = pcd.voxel_down_sample_and_trace(voxel_size=0.004, min_bound=pcd.get_min_bound(), max_bound=pcd.get_max_bound())[0]
        o3d.io.write_point_cloud(save_path, pcd_downsampled)

        return point_map, point_map_by_unprojection  # [B, F, H, W, 3]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args):
    # Set random seed
    set_seed(args.seed)
    
    args.output_dir = Path(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[Output Dir] {args.output_dir}")

    model = GeometryAdapterTrainer(
        args=args,
        learning_rate=args.learning_rate,
        recon_loss_weight=args.recon_loss_weight,
        similarity_loss_weight=args.similarity_loss_weight,
        align_kl_loss_weight=args.align_kl_loss_weight,
        diffusers_checkpoint_path=args.checkpoint_path,
    )

    if args.report_to == "tensorboard":
        logger = pl.pytorch.loggers.TensorBoardLogger(save_dir=args.output_dir)
    elif args.report_to == "wandb":
        logger = pl.pytorch.loggers.WandbLogger(
            save_dir=args.output_dir, 
            name=args.output_dir.name,
            project="gen3r_geo_adapter",
        )

    # Configure distributed training strategy
    if args.distributed:
        strategy = DDPStrategy(
            find_unused_parameters=True,
            static_graph=True,
            gradient_as_bucket_view=True,
            bucket_cap_mb=100,
            broadcast_buffers=True,
        )
    elif args.use_deepspeed:
        strategy = DeepSpeedStrategy(
            stage=2,
            pin_memory=True,
        )
    else:
        strategy = "auto"

    trainer = pl.Trainer(
        max_epochs=args.num_train_epochs,
        accelerator="gpu",
        devices="auto",
        num_nodes=1,
        precision="bf16-mixed",
        strategy=strategy,
        default_root_dir=args.output_dir,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval='step'),
        ],
        logger=logger,
        log_every_n_steps=1,
        val_check_interval=0.2,  # Validate every 0.2 epochs
        num_sanity_val_steps=1,
        sync_batchnorm=True,  # Add this to ensure batch norm stats are synchronized
        gradient_clip_val=0.5,  # Add gradient clipping to prevent exploding gradients
    )

    # If resuming with a Lightning .ckpt, run one validation pass before fit so state loads for eval
    # if args.resume_from:
    #     trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.total.completed = int(args.resume_from.split('=')[-1].split('.')[0])
    #     trainer.validate(model, ckpt_path=args.resume_from)

    trainer.fit(model, ckpt_path=args.resume_from)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to the state file to resume from.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the image vae checkpoint file.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=25,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--recon_loss_weight",
        type=float,
        default=0.5,  # 0.5
        help="Reconstruction loss weight.",
    )
    parser.add_argument(
        "--similarity_loss_weight",
        type=float,
        default=0.1,
        help="Cosine similarity loss weight.",
    )
    parser.add_argument(
        "--align_kl_loss_weight",
        type=float,
        default=2.,  # 0.05,
        help="Align KL loss weight.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training",
    )
    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Enable deepspeed training",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=8,
        help="Number of frames to sample from each video.",
    )
    parser.add_argument(
        "--video_sample_stride",
        type=int,
        default=1,
        help="Stride of frames to sample from each video.",
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=518,
        help="Size of the video sample.",
    )
    parser.add_argument(
        "--vggt_path",
        type=str,
        default=None,
        help="Path to the VGG-T checkpoint.",
    )
    parser.add_argument(
        "--wan_vae_path",
        type=str,
        default=None,
        help="Path to the wan_vae checkpoint file.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help="Report to tensorboard or wandb.",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train(args)