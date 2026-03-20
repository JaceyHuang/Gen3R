"""Modified from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""
#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import gc
import logging
import math
import os
import traceback
import random
import shutil
import sys
import time

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
from accelerate import Accelerator, DeepSpeedPlugin, DistributedType, init_empty_weights
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import (
    ProjectConfiguration, 
    set_seed,
    gather_object,
    DummyOptim,
    DummyScheduler,
)
from omegaconf import OmegaConf
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    EMAModel,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3
)
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from einops import rearrange
from packaging import version
from pathlib import Path
from safetensors.torch import load_file

from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils import ContextManagers

import datasets

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from gen3r.data import BalancedDatasetWithResizeCrop
from gen3r.models import (
    VGGT, 
    WanT5EncoderModel, 
    CLIPModel, 
    WanTransformer3DModel,
    AutoencoderKLWan, 
    GeometryAdapter, 
)
from gen3r.models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from gen3r.models.vggt.utils.geometry import unproject_depth_map_to_point_map
from gen3r.pipeline import (
    Gen3RPipeline,
    retrieve_timesteps
)
from gen3r.utils.common_utils import (
    save_videos_grid,
    string_to_filename,
    colorize_depth_map,
    downsample_and_save_pointcloud,
)
from gen3r.utils.data_utils import (
    load_prompts, 
    select_prompts,
    load_cameras,
    load_extracted_videos,
    preprocess_cameras,
    preprocess_extracted_video_with_resize_crop,
    compute_rays,
    create_dataset_config,
)
from gen3r.utils.discrete_sampler import DiscreteSampling
from gen3r.utils.loss_utils import compute_camera_loss, compute_depth_loss, compute_point_loss


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")


if is_wandb_available():
    import wandb


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value


def load_partial_state_dict(model, state_dict):
    for k, v in state_dict.items():
        if k.startswith('model.'):
            k = k[6:]
            if k.startswith('main_'):
                k = k[5:]
            try:
                model.state_dict()[k].data.copy_(v)
            except Exception as e:
                print(f"Skipping {k} for {e}.")


def decode_vggt_tokens(vggt, aggregated_token_list, fake_frames, colorize_depth=True, require_unproject=True):
    pose_enc = vggt.camera_head(aggregated_token_list)[-1]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, fake_frames.shape[-2:])
    depth_maps, _ = vggt.depth_head(aggregated_token_list, fake_frames, 5)  # [B, T, H, W, 1]
    point_maps, _ = vggt.point_head(aggregated_token_list, fake_frames, 5)  # [B, T, H, W, 3]
    if require_unproject:
        point_map_by_unprojection = torch.from_numpy(unproject_depth_map_to_point_map(
            depth_maps.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0))).unsqueeze(0)  # [B, F, H, W, 3]
    else:
        point_map_by_unprojection = None
    if colorize_depth:
        depth_maps = colorize_depth_map(depth_maps[0])[None, ...]  # [F, H, W, 1] -> [B, F, H, W, 3]

    return pose_enc, depth_maps, point_maps, point_map_by_unprojection


def get_aggregated_token_list(latents: torch.Tensor, patch_size: int) -> torch.Tensor:

    B, T, H, W, D = latents.shape
    D = D // 5

    frames = torch.zeros(
        (B, T, 3, H*patch_size, W*patch_size), device=latents.device, dtype=latents.dtype)
    cam_register_tokens = latents[..., -D:].mean(dim=(2, 3)).unsqueeze(2).repeat(1, 1, 5, 1)  # [B, T, 5, 2048]
    aggregated_token_list = [
        torch.cat([cam_register_tokens, latents[..., :D].reshape(B, T, -1, D)], dim=2),
        torch.cat([cam_register_tokens, latents[..., D:2*D].reshape(B, T, -1, D)], dim=2),
        torch.cat([cam_register_tokens, latents[..., 2*D:3*D].reshape(B, T, -1, D)], dim=2),
        torch.cat([cam_register_tokens, latents[..., 3*D:4*D].reshape(B, T, -1, D)], dim=2),
    ]

    return aggregated_token_list, frames


def prepare_validation_data(args, dataset_names, dataset_weights, num_selections):
    all_validation_prompts = load_prompts(os.path.join(args.train_data_dir, 'test_captions_paths.txt'))
    if Path(all_validation_prompts[0]).is_file():
        validation_prompts, selected_indices = select_prompts(all_validation_prompts, dataset_names, dataset_weights, num_selections)

    all_validation_cameras = load_cameras(os.path.join(args.train_data_dir, 'test_cameras_paths.txt'))
    validation_cameras = [all_validation_cameras[i] for i in selected_indices]

    all_validation_videos = load_extracted_videos(os.path.join(args.train_data_dir, 'test_videos_dirs.txt'))
    validation_videos = [all_validation_videos[i] for i in selected_indices]
    
    return validation_prompts, validation_cameras, validation_videos, selected_indices


def log_validation(
    args, config, vggt, geo_adapter, wan_vae, text_encoder, tokenizer, transformer3d_or_path, clip_image_encoder, accelerator, 
    weight_dtype, global_step, num_validations=None, dataset_names=None, dataset_weights=None
):
    try:
        logger.info("Running validation... ")
        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )
        if isinstance(transformer3d_or_path, str):
            transformer3d = WanTransformer3DModel.from_pretrained(transformer3d_or_path)
        else:
            transformer3d = transformer3d_or_path
        pipeline = Gen3RPipeline(
            vggt=accelerator.unwrap_model(vggt).to(weight_dtype), 
            geo_adapter=accelerator.unwrap_model(geo_adapter).to(weight_dtype),
            wan_vae=accelerator.unwrap_model(wan_vae).to(weight_dtype),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            transformer=accelerator.unwrap_model(transformer3d) if not isinstance(transformer3d_or_path, str) else transformer3d,
            scheduler=scheduler,
            clip_image_encoder=clip_image_encoder,
        )
        pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)

        if args.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

        validation_prompts, validation_cameras, validation_videos, selected_indices = prepare_validation_data(
            args, dataset_names, dataset_weights, 
            num_selections=max(accelerator.num_processes//2, 1) if num_validations is None else num_validations
        )
        if num_validations is None:
            num_validations = len(validation_prompts)

        all_artifacts = []
        for i in range(num_validations):
            if i % accelerator.num_processes != accelerator.process_index:
                continue

            with torch.no_grad():
                with torch.autocast('cuda', dtype=weight_dtype):
                    interval = args.validate_video_sample_stride
                    Ks, c2ws, frame_paths = preprocess_cameras(
                        validation_cameras[i], 
                        args.video_sample_n_frames, 
                        args.video_sample_size,
                        args.video_sample_size,
                        interval=interval,
                        fix_start_frame=0,
                    )
                    frames, invalid_indices, num_first_repeats, num_last_repeats = preprocess_extracted_video_with_resize_crop(
                        validation_videos[i],
                        args.video_sample_size,
                        args.video_sample_size,
                        frame_paths, 
                    )  # [F, 3, H, W]
                    frames = (frames / 255.).clamp(0, 1)  # normalize to [0, 1]
                    if len(invalid_indices) > 0:
                        c2ws = [c2ws[i] for i in range(len(c2ws)) if i not in invalid_indices]
                        Ks = [Ks[i] for i in range(len(Ks)) if i not in invalid_indices]
                        if num_first_repeats > 0:
                            c2ws = [c2ws[0]] * num_first_repeats + c2ws
                            Ks = [Ks[0]] * num_first_repeats + Ks
                        if num_last_repeats > 0:
                            c2ws = c2ws + [c2ws[-1]] * num_last_repeats
                            Ks = Ks + [Ks[-1]] * num_last_repeats
                        c2ws = torch.stack(c2ws)
                        Ks = torch.stack(Ks)
                    
                    # compute plucker ray embeddings
                    rays_o, rays_d = compute_rays(
                        c2ws.to(accelerator.device), 
                        Ks.to(accelerator.device), 
                        h=args.video_sample_size, 
                        w=args.video_sample_size,
                        device=accelerator.device
                    )  # [F, 3, H, W]
                    o_cross_d = torch.cross(rays_o, rays_d, dim=1)
                    plucker_embeddings = torch.cat([o_cross_d, rays_d], dim=1)  # [F, 6, H, W]

                    drop_camera_prob = random.random()
                    if drop_camera_prob < 0.5:
                        plucker_embeddings = torch.zeros_like(plucker_embeddings).to(accelerator.device)
                        camera_str = 'wo_camera'
                    else:
                        camera_str = 'w_camera'

                    # sample control images
                    if i % 3 == 0:  # 1view setting
                        control_images = frames[None, [0]].float()
                        control_str = '1view'
                    elif i % 3 == 1:  # 2view setting
                        control_images = frames[None, [0, -1]].float()
                        control_str = '2view'
                    else:  # allview setting
                        control_images = frames[None, :].float()
                        control_str = 'allview'

                    # run pipeline
                    sample = pipeline(
                        prompt = validation_prompts[i],
                        control_cameras = plucker_embeddings[None, ...],  # [B, F, 6, H, W]
                        control_images = control_images,
                        num_frames = args.video_sample_n_frames,
                        negative_prompt = "bad detailed",
                        height      = args.video_sample_size,
                        width       = args.video_sample_size,
                        generator   = generator,
                        guidance_scale = 5,
                        return_dict = True,
                        min_max_depth_mask = True,
                    )
                    artifacts = {
                        'rgbs': sample.rgbs,  # [B, F, H, W, 3]
                        'depth_maps': sample.depth_maps,  # [B, F, H, W, 1]
                        "pcds": sample.pcds,  # [B, F, H, W, 3]
                        'point_masks': sample.point_masks,  # [B, F, H, W]
                        'cameras': sample.cameras,  # [B, F, 3, 4], [B, F, 3, 3]
                    }
                    
                    videos = []
                    for key, value in artifacts.items():
                        
                        if key == "rgbs" and value is not None:
                            rgb = rearrange(value, "b f h w c -> b c f h w").float().cpu()  
                            videos.append(rgb)

                        elif key == "depth_maps":
                            # colorize the depth map
                            depth = colorize_depth_map(value[0])[None, ...]  # [B, F, H, W, 1] -> [B, F, H, W, 3]
                            depth = rearrange(depth, "b f h w c -> b c f h w").cpu()
                            videos.append(depth)

                        elif key == "pcds":
                            filename = f"pcds_gen_id_{selected_indices[i]}_{string_to_filename(validation_prompts[i])[:50]}.ply"
                            filename = os.path.join(args.output_dir, "validations", f"sample-{global_step}", f'rank_{accelerator.process_index}_{camera_str}_{control_str}', filename)
                            os.makedirs(os.path.dirname(filename), exist_ok=True)
                            downsample_and_save_pointcloud(filename, value[0], artifacts['rgbs'][0])

                    if len(videos) > 0:
                        combined_latents, vggt_latents, wan_latents = batch_encode_latents(
                            frames.unsqueeze(0).to(accelerator.device), vggt, geo_adapter, wan_vae)  # [B, 16, f, 70, 140], [B, 16, f, 70, 70]
                        adapted_vggt_latents = combined_latents[..., combined_latents.shape[-1]//2:]  # [B, 16, f, 70, 70]

                        # 1. reconstructed pseudo gt from geometry adapter
                        decoded_vggt_latents = geo_adapter.decode(adapted_vggt_latents).sample  # [B, 10240, F, 40, 40]
                        aggregated_token_list, fake_frames = get_aggregated_token_list(
                            rearrange(decoded_vggt_latents, 'b c f h w -> b f h w c'), 
                            vggt.aggregator.patch_size
                        )
                        
                        reconstructed_rgbs = wan_vae.decode(wan_latents).sample  # [B, 3, F, H, W]
                        reconstructed_rgbs = rearrange((reconstructed_rgbs / 2 + 0.5).clamp(0, 1), 'b c f h w -> b f h w c')  # [B, F, H, W, 3]
                        _, reconstructed_depth_maps, _, reconstructed_point_map_by_unprojection = decode_vggt_tokens(
                            vggt, aggregated_token_list, fake_frames, colorize_depth=True, require_unproject=True
                        )

                        filename = f"pcds_recon_id_{selected_indices[i]}_{string_to_filename(validation_prompts[i])[:50]}.ply"
                        filename_pcd = os.path.join(args.output_dir, "validations", f"sample-{global_step}", f'rank_{accelerator.process_index}_{camera_str}_{control_str}', filename)
                        os.makedirs(os.path.dirname(filename_pcd), exist_ok=True)
                        downsample_and_save_pointcloud(filename_pcd, reconstructed_point_map_by_unprojection, reconstructed_rgbs, filter_outliers=False)

                        # 2. pseudo gt from vggt
                        aggregated_token_list, fake_frames = get_aggregated_token_list(
                            rearrange(vggt_latents, 'b c f h w -> b f h w c'), 
                            vggt.aggregator.patch_size
                        )
                        _, depth_maps, _, point_map_by_unprojection = decode_vggt_tokens(
                            vggt, aggregated_token_list, fake_frames, colorize_depth=True, require_unproject=True
                        )

                        filename = f"pcds_vggt_id_{selected_indices[i]}_{string_to_filename(validation_prompts[i])[:50]}.ply"
                        filename_pcd = os.path.join(args.output_dir, "validations", f"sample-{global_step}", f'rank_{accelerator.process_index}_{camera_str}_{control_str}', filename)
                        os.makedirs(os.path.dirname(filename_pcd), exist_ok=True)
                        downsample_and_save_pointcloud(filename_pcd, point_map_by_unprojection, rearrange(frames, 'f c h w-> (f h w) c'), filter_outliers=False)

                        # 3. video contains all rgbs and depths
                        # vggt pseudo gt
                        vggt_video = torch.cat([frames[None].permute(0, 2, 1, 3, 4), depth_maps.permute(0, 4, 1, 2, 3).cpu()], dim=-1)
                        # geometry adapter reconstructed pseudo gt
                        reconstructed_video = torch.cat([reconstructed_rgbs.permute(0, 4, 1, 2, 3).cpu(), reconstructed_depth_maps.permute(0, 4, 1, 2, 3).cpu()], dim=-1)
                        # gt rgb
                        gt_video = torch.cat([frames[None].permute(0, 2, 1, 3, 4), torch.ones_like(frames[None]).permute(0, 2, 1, 3, 4)], dim=-1)

                        videos = torch.cat(
                            [torch.cat([gt_video, vggt_video], dim=-1).cpu(), 
                             torch.cat([reconstructed_video, torch.cat(videos, dim=-1)], dim=-1).cpu()], dim=-2
                        )

                        filename = f"video_id_{selected_indices[i]}_{string_to_filename(validation_prompts[i])[:50]}.mp4"
                        filename_video = os.path.join(args.output_dir, "validations", f"sample-{global_step}", f'rank_{accelerator.process_index}_{camera_str}_{control_str}', filename)
                        logger.debug(f"Saving video to {filename_video}")
                        os.makedirs(os.path.dirname(filename_video), exist_ok=True)
                        save_videos_grid(videos, filename_video, rescale=False)
                        
                        value = wandb.Video(filename_video, caption=f"id: {selected_indices[i]}, {camera_str}, {control_str}, prompt: {validation_prompts[i]}")
                        all_artifacts.append(value)

        # Gather all artifacts from all processes
        gathered_artifacts = gather_object(all_artifacts)
        logger.info(f"Gathered {len(gathered_artifacts)} artifacts from all processes")
        
        # Only log on main process
        if accelerator.is_main_process:
            tracker_key = "validation"
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    video_artifacts = [
                        artifact for artifact in gathered_artifacts if isinstance(artifact, wandb.Video)
                    ]
                    tracker.log(
                        {
                            tracker_key: {"videos": video_artifacts},
                        },
                        step=global_step,
                    )
            
            # save pipeline
            if args.checkpoints_total_limit is not None:
                pipelines = os.listdir(os.path.join(args.output_dir, "pipelines"))
                pipelines = [d for d in pipelines if d.startswith("pipeline")]
                pipelines = sorted(pipelines, key=lambda x: int(x.split("-")[1]))

                # before we save the new pipeline, we need to have at _most_ `checkpoints_total_limit - 1` pipelines
                if len(pipelines) >= args.checkpoints_total_limit:
                    num_to_remove = len(pipelines) - args.checkpoints_total_limit + 1
                    removing_pipelines = pipelines[0:num_to_remove]

                    logger.info(
                        f"{len(pipelines)} pipelines already exist, removing {len(removing_pipelines)} pipelines"
                    )
                    logger.info(f"removing pipelines: {', '.join(removing_pipelines)}")

                    for removing_pipeline in removing_pipelines:
                        removing_pipeline = os.path.join(args.output_dir, "pipelines", removing_pipeline)
                        shutil.rmtree(removing_pipeline)
            
            pipeline.save_pretrained(os.path.join(args.output_dir, "pipelines", f'pipeline-{global_step}'))
            logger.info(f"Saved pipeline to {os.path.join(args.output_dir, 'pipelines', f'pipeline-{global_step}')}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Eval error with info {e}")
        traceback.print_exc()


def batch_encode_latents(pixel_values, vggt, geo_adapter, wan_vae):  # [B, F, C, H, W]
    intermediate_layer_idx = vggt.depth_head.intermediate_layer_idx
    
    # list of 24 tensors with shape [B, F, (H/14)*(W/14)+ps_idx, 2C]
    aggregated_token_list, ps_idx = vggt.aggregator(pixel_values)
    h = int(math.sqrt(aggregated_token_list[0].shape[-2]-ps_idx))  # 40
    w = int(math.sqrt(aggregated_token_list[0].shape[-2]-ps_idx))  # 40
    
    # save camera and register tokens, [B, F, 1, D]
    cam_tokens = aggregated_token_list[-1][:, :, :1]
    # broadcast the camera tokens to [B, F, h, w, D]
    cam_tokens = cam_tokens.unsqueeze(2).repeat(1, 1, h, w, 1)
    
    # convert to 4 * [B, F, 40, 40, C] and ensure contiguous layout
    vggt_tokens_list = [
        rearrange(aggregated_token_list[i][:, :, ps_idx:], 'b f (h w) c -> b f h w c', h=h, w=w).contiguous()
        for i in intermediate_layer_idx
    ] 
    vggt_tokens_list.append(cam_tokens)  # 5 * [B, F, 40, 40, C]
    # turn the tokens into size [B, F, 40, 40, 5C]
    vggt_latents = rearrange(torch.cat(vggt_tokens_list, dim=-1), 'b f h w c -> b c f h w')  # [B, 10240, F, 40, 40]
    
    # adapt the tokens
    adapted_vggt_latents = geo_adapter.encode(vggt_latents).latent_dist.sample()  # [B, 16, f, 70, 70]

    # encode the images with wan_vae
    pixel_values = rearrange((pixel_values * 2 - 1).clamp(-1, 1), 'b f c h w -> b c f h w')
    wan_latents = wan_vae.encode(pixel_values).latent_dist.sample()  # [B, 16, f, 70, 70]

    combined_latents = torch.cat([wan_latents, adapted_vggt_latents], dim=-1)  # [B, 16, f, 70, 140]
    
    return combined_latents, vggt_latents, wan_latents


def batch_encode_control_latents(pixel_values, vggt, geo_adapter, wan_vae, clip_image_encoder, accelerator):
    B, F = pixel_values.shape[:2]
    
    control_images = torch.zeros_like(pixel_values)  # [1, F, C, H, W]
    mode_prob = random.random()
    if mode_prob < 1/3:  # use the start frame as control image
        control_index = [0]
    elif mode_prob > 2/3:  # use the start-end frames as control images
        control_index = [0, -1]
    else:  # use all frames as control images
        control_index = [i for i in range(F)]
    control_images[:, control_index] = pixel_values[:, control_index]

    # encode control latents
    control_latents = batch_encode_latents(control_images, vggt, geo_adapter, wan_vae)[0]  # [1, 16, f, 70, 140]

    # prepare masks
    h, w = control_latents.shape[-2:]
    masks = torch.zeros(B, F, h, w).to(accelerator.device, control_latents.dtype)
    masks[:, control_index, :, :w//2] = 1  # [1, F, 70, 140]
    masks[:, control_index, :, w//2:] = 0
    masks = torch.cat([
        torch.repeat_interleave(masks[:, :1], repeats=4, dim=1), masks[:, 1:]], dim=1)  # [1, F+3, 70, 140]
    masks = masks.view(B, (F+3) // 4, 4, h, w).contiguous().transpose(1, 2)  # [1, 4, f, 70, 140]
    
    # concat mask to control latents
    control_latents = torch.cat([control_latents, masks], dim=1)  # [1, 20, f, 70, 140]

    # encode clip context
    clip_context = []
    for index in control_index[:1]:
        clip_image = Image.fromarray((control_images[:, index].squeeze() * 255).permute(1, 2, 0).float().cpu().numpy().astype(np.uint8))
        clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(accelerator.device, control_latents.dtype)  # in [-1, 1]
        clip_context.append(clip_image_encoder([clip_image[:, None, :, :]]))  # [1, 257, 1280]
    clip_context = torch.cat(clip_context)

    return control_latents, clip_context


def get_noise_pred(
    transformer3d, 
    noisy_latent,  # [B, 16, f, 70, 140]
    plucker_embeds,  # [B, 24, f, H, W]
    t, 
    prompt_embeds, 
    seq_len=None, 
    control_latents=None,  # [B, 20, f, 70, 140]
    clip_context=None,
):
    noise_pred = transformer3d(
        x=noisy_latent,
        context=prompt_embeds,
        t=t.expand(noisy_latent.shape[0]),
        seq_len=seq_len,
        y=control_latents,
        y_camera=plucker_embeds, 
        full_ref=None,
        clip_fea=clip_context,
    )  # [B, 16, f, 70, 140]
    return noise_pred


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_wan_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--vggt_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained vggt model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--geo_adapter_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained latent compressor model.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. "
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--use_came",
        action="store_true",
        help="whether to use came",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_wan_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_model_info", action="store_true", help="Whether or not to report more info about model (such as norm, grad)."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_and_validation_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--checkpointing_and_validation_steps",
        type=int,
        default=2000,
        help="Run checkpointing and validation every X steps. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="gen3r_dit",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    
    parser.add_argument(
        "--uniform_sampling", action="store_true", help="Whether or not to use uniform_sampling."
    )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Run train_sampling_steps.",
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=512,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--video_sample_stride",
        type=int,
        default=4,
        help="Sample stride of the video.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=17,
        help="Num frame of video.",
    )
    parser.add_argument(
        "--validate_video_sample_stride",
        type=int,
        default=1,
        help="Sample stride of the video for validation.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=(
            "The config of the model in training."
        ),
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other transformers, input its path."),
    )
    parser.add_argument(
        '--trainable_modules', 
        nargs='+', 
        help='Enter a list of trainable modules'
    )
    parser.add_argument(
        '--trainable_modules_low_learning_rate', 
        nargs='+', 
        default=[],
        help='Enter a list of trainable modules with lower learning rate'
    )
    parser.add_argument(
        '--tokenizer_max_length', 
        type=int,
        default=512,
        help='Max length of tokenizer'
    )
    parser.add_argument(
        "--abnormal_norm_clip_start",
        type=int,
        default=2000,
        help=(
            'When do we start doing additional processing on abnormal gradients. '
        ),
    )
    parser.add_argument(
        "--initial_grad_norm_ratio",
        type=int,
        default=2,
        help=(
            'The initial gradient is relative to the multiple of the max_grad_norm. '
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--sanity_check", action="store_true", help="Whether to perform sanity check."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    
    print(f"Output directory: {args.output_dir}")
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    os.environ["WANDB_DIR"] = str(logging_dir)
    os.environ["WANDB_CACHE_DIR"] = str(logging_dir)  # Set cache directory to the same location

    config = OmegaConf.load(args.config_path)
    
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    use_deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() in ("1", "true", "yes")

    if use_deepspeed:
        zero_stage = 2
        hf_ds_config = {
            'bf16': {
                'enabled': True
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": "auto",
                    "weight_decay": "auto",
                }
            },
            "scheduler": {
                "type": "WarmupCosineLR",
                "params": {
                    "warmup_num_steps": "auto",
                    "warmup_type": "linear",
                    "total_num_steps": "auto",
                }
            },
            'train_micro_batch_size_per_gpu': args.train_batch_size,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'gradient_clipping': 0.,
            'zero_optimization': {
                'stage': zero_stage,
                'overlap_comm': True,
                'contiguous_gradients': True,
                'offload_optimizer': {
                    'device': 'cpu',
                    'pin_memory': True
                },
            },
        }
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=hf_ds_config,
        )
        print(f"Using DeepSpeed Zero stage: {zero_stage}")
    else:
        zero_stage = 0
        deepspeed_plugin = None
        print("DeepSpeed is not enabled.")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        deepspeed_plugin=deepspeed_plugin,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        torch_rng = None
    # print(f"Init rng with seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths (geo_adapter, non-lora text_encoder and non-lora transformer3d) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Load scheduler, tokenizer and models.
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_wan_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()), torch.device('cpu'):
        # Get Text encoder
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(args.pretrained_wan_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
        )
        text_encoder = text_encoder.eval().cpu()

        # Get Wan VAE
        wan_vae = AutoencoderKLWan.from_pretrained(
            os.path.join(args.pretrained_wan_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        )
        wan_vae = wan_vae.eval().cpu()
        
        # Get VGGT
        vggt = VGGT.from_pretrained(args.vggt_path).cpu()
        vggt = vggt.eval().cpu()

        # Get Geometry Adapter
        geo_adapter = GeometryAdapter.from_pretrained(
            args.geo_adapter_path,
            additional_kwargs={
                "latent_channels": 16,
                "hidden_dim": 128,
                "input_dim": 10240,
                "output_dim": 10240,
                "resample_scale": 10,
                "temporal_compression_ratio": 4,
                "spatial_compression_ratio": 8,
            }
        ).cpu()
        geo_adapter = geo_adapter.eval().cpu()
            
    # Get Transformer
    transformer3d = WanTransformer3DModel.from_pretrained(
        os.path.join(args.pretrained_wan_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        torch_dtype=torch.float32 if weight_dtype != torch.bfloat16 else weight_dtype,
    ).cpu()
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(args.pretrained_wan_name_or_path, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    )
    clip_image_encoder = clip_image_encoder.eval().cpu()

    # Freeze other modules and set transformer3d to trainable
    text_encoder.requires_grad_(False)
    wan_vae.requires_grad_(False)
    vggt.requires_grad_(False)
    geo_adapter.requires_grad_(False)
    transformer3d.requires_grad_(False)
    clip_image_encoder.requires_grad_(False)

    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        # print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
    
    # A good trainable modules is showed below now.
    # For 3D Patch: trainable_modules = ['ff.net', 'pos_embed', 'attn2', 'proj_out', 'timepositionalencoding', 'h_position', 'w_position']
    # For 2D Patch: trainable_modules = ['ff.net', 'attn2', 'timepositionalencoding', 'h_position', 'w_position']
    transformer3d.train()
    # if accelerator.is_main_process:
    #     accelerator.print(
    #         f"Trainable modules '{args.trainable_modules}'."
    #     )
    for name, param in transformer3d.named_parameters():
        for trainable_module_name in args.trainable_modules + args.trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    # Create EMA for the transformer3d.
    if args.use_ema:
        if zero_stage == 3:
            raise NotImplementedError("FSDP does not support EMA.")

        ema_transformer3d = WanTransformer3DModel.from_pretrained(
            os.path.join(args.pretrained_wan_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            torch_dtype=torch.float32 if weight_dtype != torch.bfloat16 else weight_dtype,
        ).cpu()
        ema_transformer3d = EMAModel(ema_transformer3d.parameters(), model_cls=WanTransformer3DModel, model_config=ema_transformer3d.config)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        if not zero_stage == 3:
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    if args.use_ema:
                        ema_transformer3d.save_pretrained(os.path.join(output_dir, "transformer_ema"))
                    models[0].save_pretrained(os.path.join(output_dir, "transformer"))
                    if weights:
                        weights.pop()

            def load_model_hook(models, input_dir):
                init_under_meta = False

                if args.use_ema:
                    ema_path = os.path.join(input_dir, "transformer_ema")
                    _, ema_kwargs = WanTransformer3DModel.load_config(ema_path, return_unused_kwargs=True)
                    load_model = WanTransformer3DModel.from_pretrained(
                        input_dir, subfolder="transformer_ema",
                        torch_dtype=torch.float32 if weight_dtype != torch.bfloat16 else weight_dtype,
                    )
                    load_model = EMAModel(load_model.parameters(), model_cls=WanTransformer3DModel, model_config=load_model.config)
                    
                    load_model.load_state_dict(ema_kwargs)
                    ema_transformer3d.load_state_dict(load_model.state_dict())
                    ema_transformer3d.to(accelerator.device)
                    del load_model

                if not accelerator.distributed_type == DistributedType.DEEPSPEED:
                    while len(models) > 0:
                        model = models.pop()
                        if isinstance(accelerator.unwrap_model(model), type(accelerator.unwrap_model(transformer3d))):
                            model = accelerator.unwrap_model(model)
                else:
                    with init_empty_weights():
                        model = WanTransformer3DModel.from_pretrained(
                            os.path.join(args.pretrained_wan_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
                            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
                        )
                        init_under_meta = True

                # load diffusers style into model
                load_model = WanTransformer3DModel.from_pretrained(
                    input_dir, subfolder="transformer",
                    torch_dtype=torch.float32 if weight_dtype != torch.bfloat16 else weight_dtype,
                )
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict(), assign=init_under_meta)
                del load_model
        
        else:
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                pass

            def load_model_hook(models, input_dir):
                pass

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif args.use_came:
        try:
            from came_pytorch import CAME
        except:
            raise ImportError(
                "Please install came_pytorch to use CAME. You can do so by running `pip install came_pytorch`"
            )

        optimizer_cls = CAME
    elif use_deepspeed:
        # optimizer_cls = deepspeed.ops.adam.FusedAdam
        optimizer_cls = DummyOptim
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))
    trainable_params_optim = [
        {'params': [], 'lr': args.learning_rate},
        {'params': [], 'lr': args.learning_rate / 2},
    ]
    in_already = []
    for name, param in transformer3d.named_parameters():
        high_lr_flag = False
        if name in in_already:
            continue
        for trainable_module_name in args.trainable_modules:
            if trainable_module_name in name:
                in_already.append(name)
                high_lr_flag = True
                trainable_params_optim[0]['params'].append(param)
                # if accelerator.is_main_process:
                #     print(f"Set {name} to lr : {args.learning_rate}")
                break
        if high_lr_flag:
            continue
        for trainable_module_name in args.trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                in_already.append(name)
                trainable_params_optim[1]['params'].append(param)
                # if accelerator.is_main_process:
                #     print(f"Set {name} to lr : {args.learning_rate / 2}")
                break

    if args.use_came:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            # weight_decay=args.adam_weight_decay,
            betas=(0.9, 0.999, 0.9999), 
            eps=(1e-30, 1e-16)
        )
    else:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Get the training dataset
    dataset_configs, dataset_names, dataset_weights = create_dataset_config(args)
    train_dataset = BalancedDatasetWithResizeCrop(
        dataset_configs=dataset_configs,
        max_num_frames=args.video_sample_n_frames,
        height=args.video_sample_size,
        width=args.video_sample_size,
        max_interval=args.video_sample_stride,
        dataset_weights=dataset_weights,
    )

    def worker_init_fn(_seed):
        _seed = _seed * 256
        def _worker_init_fn(worker_id):
            # print(f"worker_init_fn with {_seed + worker_id}")
            np.random.seed(_seed + worker_id)
            random.seed(_seed + worker_id)
        return _worker_init_fn
        
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        persistent_workers=True if args.dataloader_num_workers != 0 else False,
        num_workers=args.dataloader_num_workers,
        worker_init_fn=worker_init_fn(args.seed + accelerator.process_index),
        shuffle=True,
        pin_memory=True,
        prefetch_factor=2 if args.dataloader_num_workers > 0 else None,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if not use_deepspeed:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer=optimizer,
            warmup_num_steps=args.lr_warmup_steps,
            total_num_steps=args.max_train_steps,
        )
    
    # Prepare everything with our `accelerator`.
    transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer3d, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_transformer3d.to(accelerator.device, dtype=weight_dtype)

    # Move text_encoder and other modules to gpu and cast to weight_dtype
    vggt.to(accelerator.device, dtype=weight_dtype)
    geo_adapter.to(accelerator.device, dtype=weight_dtype)
    wan_vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    clip_image_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    args.abnormal_norm_clip_start = int(args.max_train_steps * 0.1)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        # tracker_config.pop("validation_prompts")
        tracker_config.pop("trainable_modules")
        tracker_config.pop("trainable_modules_low_learning_rate")
        if args.report_to == "wandb":
            tracker_config['wandb'] = {'name': args.output_dir.split('/')[-1]}
        accelerator.init_trackers(args.tracker_project_name, init_kwargs=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Abnormal norm clip start = {args.abnormal_norm_clip_start}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            # path = os.path.basename(args.resume_from_checkpoint)
            path = args.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = os.path.join(args.output_dir, dirs[-1]) if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            global_step = int(path.split("-")[-1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            print(f"Get first_epoch = {first_epoch}.")

            accelerator.print(f"Resuming from checkpoint {path}")
            torch.cuda.empty_cache()
            time.sleep(accelerator.process_index * 2)  # peak shifting to avoid OOM
            if not use_deepspeed:
                accelerator.load_state(path, map_location="cpu")
            else:
                accelerator.load_state(path)
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    for epoch in range(first_epoch, args.num_train_epochs):
        avg_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            
            # Data batch sanity check
            if epoch == first_epoch and step == 0 and accelerator.is_main_process and args.sanity_check:    
                pixel_values, texts = batch['pixel_values'][:1].cpu(), batch['text'][:1]
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/data_loader_check.gif", rescale=False)

            with accelerator.accumulate(transformer3d):
                # Convert images to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)  # [B, F, C, H, W]
                bsz, num_frames, _, height, width = pixel_values.size()

                with torch.no_grad():
                    combined_latents_list, vggt_latents_list, control_latents_list, clip_context_list = [], [], [], []
                    for i in range(pixel_values.size()[0]):
                        combined_latents, vggt_latents, _ = batch_encode_latents(
                            pixel_values[i].unsqueeze(0), vggt, geo_adapter, wan_vae
                        )  # [B, 16, f, 70, 140]
                        control_latents, clip_context = batch_encode_control_latents(
                            pixel_values[i].unsqueeze(0), vggt, geo_adapter, wan_vae, clip_image_encoder, accelerator
                        )  # [B, 20, f, 70, 140]
                        combined_latents_list.append(combined_latents)
                        vggt_latents_list.append(vggt_latents)
                        control_latents_list.append(control_latents)
                        clip_context_list.append(clip_context)
                    combined_latents = torch.cat(combined_latents_list, dim=0)
                    vggt_latents = torch.cat(vggt_latents_list, dim=0)
                    control_latents = torch.cat(control_latents_list, dim=0)
                    clip_context = torch.cat(clip_context_list, dim=0)

                prompt_embeddings = []
                with torch.no_grad():
                    for i in range(len(batch['text'])):
                        # randomly drop the prompt
                        if random.random() < 0.2:
                            batch['text'][i] = ""
                        prompt_ids = tokenizer(
                            batch['text'][i], padding="max_length", max_length=args.tokenizer_max_length, 
                            truncation=True, add_special_tokens=True, return_tensors="pt"
                        )
                        text_input_ids = prompt_ids.input_ids
                        prompt_attention_mask = prompt_ids.attention_mask

                        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                        prompt_embeds = text_encoder(
                            text_input_ids.to(accelerator.device), attention_mask=prompt_attention_mask.to(accelerator.device))[0]
                        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
                        prompt_embeds = torch.stack(
                            [torch.cat([u, u.new_zeros(args.tokenizer_max_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
                        )
                        prompt_embeddings.append(prompt_embeds)
                    
                    prompt_embeds = torch.cat(prompt_embeddings, dim=0)

                plucker_embeds = []
                for i in range(len(batch['c2ws'])):  # [B F 4 4]
                    # compute plucker ray embeddings
                    rays_o, rays_d = compute_rays(
                        batch['c2ws'][i].to(accelerator.device), 
                        batch['Ks'][i].to(accelerator.device), 
                        h=args.video_sample_size, 
                        w=args.video_sample_size,
                        device=accelerator.device
                    )  # [F, 3, H, W]
                    o_cross_d = torch.cross(rays_o, rays_d, dim=1)
                    plucker_embed = torch.cat([o_cross_d, rays_d], dim=1).to(combined_latents.dtype)  # [F, 6, H, W]
                    plucker_embeds.append(plucker_embed)
                
                plucker_embeds = torch.stack(plucker_embeds, dim=0)  # [B, F, 6, H, W]
                plucker_embeds = rearrange(plucker_embeds, 'b f c h w -> b c f h w')  # [B, 6, F, H, W]

                plucker_embeds = torch.cat(
                    [torch.repeat_interleave(plucker_embeds[:, :, 0:1], repeats=4, dim=2), plucker_embeds[:, :, 1:]], dim=2
                ).transpose(1, 2).contiguous()  # [B, F+3, 6, H, W]
                plucker_embeds = plucker_embeds.view(bsz, (num_frames + 3) // 4, 4, plucker_embeds.shape[2], height, width)  # [B, (F+3)//4, 4, 6, H, W]
                plucker_embeds = plucker_embeds.transpose(2, 3).contiguous()  # [B, (F+3)//4, 6, 4, H, W]
                plucker_embeds = plucker_embeds.view(bsz, (num_frames + 3) // 4, plucker_embeds.shape[2] * 4, height, width)  # [B, (F+3)//4, 24, H, W]
                plucker_embeds = plucker_embeds.transpose(1, 2)  # [B, 24, (F+3)//4, H, W], that is [B, 24, f, H, W]
                # duplicate in width dimension
                plucker_embeds = torch.cat([plucker_embeds, plucker_embeds], dim=-1)  # [B, 24, f, H, 2*W]
                
                # randomly drop the plucker
                if random.random() < 0.5:
                    plucker_embeds = torch.zeros_like(plucker_embeds)

                # sanity_check
                if epoch == first_epoch and step == 0 and accelerator.is_main_process and args.sanity_check:
                    with torch.no_grad():
                        sanity_check_path = os.path.join(args.output_dir, "sanity_check")
                        os.makedirs(sanity_check_path, exist_ok=True)
                        
                        # vggt, wan_vae and geo_adapter sanity check
                        aggregated_token_list, frames = get_aggregated_token_list(
                            rearrange(vggt_latents[:1], 'b c f h w -> b f h w c'), 
                            vggt.aggregator.patch_size
                        )
                        _, depth_map, _, point_map_by_unprojection = decode_vggt_tokens(vggt, aggregated_token_list, frames)

                        adapted_vggt_latents = combined_latents[:1].chunk(2, dim=-1)[1].detach().clone()  # [B, 16, f, 70, 70]
                        decoded_vggt_latents = geo_adapter.decode(adapted_vggt_latents).sample  # [B, 10240, F, 40, 40]
                        aggregated_token_list, frames = get_aggregated_token_list(
                            rearrange(decoded_vggt_latents, 'b c f h w -> b f h w c'), 
                            vggt.aggregator.patch_size
                        )
                        _, reconstructed_depth_map, _, reconstructed_point_map_by_unprojection = decode_vggt_tokens(vggt, aggregated_token_list, frames)
                        
                        wan_latents = combined_latents[:1].chunk(2, dim=-1)[0].detach().clone()  # [1, 16, f, 70, 70]
                        reconstructed_rgb = wan_vae.decode(wan_latents).sample  # [1, 3, F, H, W]
                        reconstructed_rgb = rearrange((reconstructed_rgb / 2 + 0.5).clamp(0, 1), 'b c f h w -> b f h w c')

                        video_filename = os.path.join(sanity_check_path, f"vggt_geo_adapter.mp4")
                        video = torch.cat([pixel_values[:1].permute(0, 1, 3, 4, 2), depth_map, reconstructed_rgb, reconstructed_depth_map], dim=-2).cpu()
                        save_videos_grid(rearrange(video, "b f h w c -> b c f h w"), video_filename, rescale=False)

                        # downsample the point map using open3d
                        for i, point_map in enumerate([point_map_by_unprojection, reconstructed_point_map_by_unprojection]):
                            colors = rearrange(reconstructed_rgb, "b f h w c -> (b f h w) c") if i == 1 else rearrange(pixel_values[:1], "b f c h w -> (b f h w) c")
                            ply_filename = os.path.join(sanity_check_path, f"pcd_geo_adapter.ply") if i == 1 else os.path.join(sanity_check_path, f"pcd_vggt.ply")
                            downsample_and_save_pointcloud(ply_filename, point_map, colors)

                        del aggregated_token_list, frames
                        torch.cuda.empty_cache()

                # Validate sanity check
                if ((epoch == first_epoch and step == 0)) and args.sanity_check:
                    if args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_transformer3d.store(transformer3d.parameters())
                        ema_transformer3d.copy_to(transformer3d.parameters()) 
                    log_validation(
                        args, config, vggt, geo_adapter, wan_vae, text_encoder, tokenizer, transformer3d, clip_image_encoder, accelerator, 
                        weight_dtype, global_step, num_validations=2, dataset_names=dataset_names, dataset_weights=dataset_weights
                    )
                    if args.use_ema:
                        # Switch back to the original transformer3d parameters.
                        ema_transformer3d.restore(transformer3d.parameters())

                _, latent_channel, latent_num_frames, latent_height, latent_width = combined_latents.size()  # [B, 16, f, 70, 140]
                noise = torch.randn(combined_latents.size(), device=combined_latents.device, generator=torch_rng, dtype=weight_dtype)

                target_shape = (latent_channel, latent_num_frames, latent_width, latent_height)
                seq_len = math.ceil(
                    (target_shape[2] * target_shape[3]) /
                    (accelerator.unwrap_model(transformer3d).config.patch_size[1] * accelerator.unwrap_model(transformer3d).config.patch_size[2]) *
                    target_shape[1]
                )

                def get_sigmas(timesteps, scheduler, n_dim=4, dtype=torch.float32):
                    sigmas = scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
                    schedule_timesteps = scheduler.timesteps.to(accelerator.device)
                    timesteps = timesteps.to(accelerator.device)
                    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

                    sigma = sigmas[step_indices].flatten()
                    while len(sigma.shape) < n_dim:
                        sigma = sigma.unsqueeze(-1)
                    return sigma

                if not args.uniform_sampling:
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler.config.num_train_timesteps).long()
                else:
                    # Sample a random timestep for each image
                    # timesteps = generate_timestep_with_lognorm(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                    # timesteps = torch.randint(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                    indices = idx_sampling(bsz, generator=torch_rng, device=accelerator.device)
                    indices = indices.long().cpu()
                timesteps = noise_scheduler.timesteps[indices].to(device=accelerator.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, noise_scheduler, n_dim=combined_latents.ndim, dtype=combined_latents.dtype)
                noisy_latents = (1.0 - sigmas) * combined_latents + sigmas * noise  # [B, 16, f, 70, 140]

                # Get target
                target = noise - combined_latents

                # Predict the noise residual
                with torch.cuda.amp.autocast(dtype=weight_dtype):
                    noise_pred = get_noise_pred(
                        transformer3d, noisy_latents, plucker_embeds, timesteps, prompt_embeds, seq_len, control_latents, clip_context
                    )  # [B, 16, f, 70, 140]
                
                def custom_mse_loss(noise_pred, target, weighting=None, threshold=50):
                    noise_pred = noise_pred.float()
                    target = target.float()
                    diff = noise_pred - target
                    mse_loss = F.mse_loss(noise_pred, target, reduction='none')
                    mask = (diff.abs() <= threshold).float()
                    masked_loss = mse_loss * mask
                    if weighting is not None:
                        masked_loss = masked_loss * weighting
                    final_loss = masked_loss.mean()
                    return final_loss
                
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                diff_loss = custom_mse_loss(noise_pred.float().contiguous(), target.float().contiguous(), weighting.float().contiguous())
                loss = diff_loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss += accelerator.reduce(loss, reduction="mean").item() / args.gradient_accumulation_steps

                # Backpropagate
                if use_deepspeed:
                    max_grad_norm = linear_decay(args.max_grad_norm * args.initial_grad_norm_ratio, args.max_grad_norm, args.abnormal_norm_clip_start, global_step)
                    if accelerator.sync_gradients:
                        accelerator.deepspeed_engine_wrapped.engine._config.gradient_clipping = max_grad_norm
                        accelerator.deepspeed_engine_wrapped.engine.optimizer.clip_grad = max_grad_norm
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if not use_deepspeed and args.report_model_info and accelerator.is_main_process:
                        max_grad_norm = linear_decay(args.max_grad_norm * args.initial_grad_norm_ratio, args.max_grad_norm, args.abnormal_norm_clip_start, global_step)
                        norm_sum_before_clip = accelerator.clip_grad_norm_(trainable_params, max_grad_norm)
                        trainable_params_grads = [p.grad for p in trainable_params if p.grad is not None]
                        norm_sum_after_clip = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in trainable_params_grads]), 2)
                        accelerator.log(
                            {
                                f'gradients/norm_sum_before_clip': norm_sum_before_clip, 
                                f'gradients/actual_max_grad_norm': max_grad_norm,
                                f'gradients/norm_sum_after_clip': norm_sum_after_clip, 
                            }, 
                            step=global_step+1,
                        )
                    elif use_deepspeed and args.report_model_info and accelerator.is_main_process:
                        norm_sum_before_clip = accelerator.deepspeed_engine_wrapped.engine.get_global_grad_norm()
                        accelerator.log(
                            {
                                f'gradients/norm_sum_before_clip': norm_sum_before_clip, 
                                f'gradients/actual_max_grad_norm': max_grad_norm,
                            }, 
                            step=global_step+1,
                        )
                    
                    if use_deepspeed and args.report_model_info:
                        if hasattr(accelerator.deepspeed_engine_wrapped.engine.optimizer, '_global_grad_norm_clipped'):
                            local_norm = accelerator.deepspeed_engine_wrapped.engine.optimizer._global_grad_norm_clipped
                            local_sq = torch.tensor(float(local_norm) ** 2, device=accelerator.device)
                            global_sq = accelerator.reduce(local_sq, reduction="sum")
                            norm_sum_after_clip = torch.sqrt(global_sq).item()
                        elif accelerator.is_main_process:
                            norm_sum_after_clip = min(norm_sum_before_clip, max_grad_norm)
                        if accelerator.is_main_process:
                            accelerator.log(
                                {
                                    f'gradients/norm_sum_after_clip': norm_sum_after_clip, 
                                }, 
                                step=global_step+1,
                            )

                # Always step the optimizer and scheduler; Accelerate/DeepSpeed handle the wrapped optimizer internally
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_transformer3d.step(transformer3d.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"loss": avg_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                avg_loss = 0.0

                if global_step % args.checkpointing_and_validation_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(os.path.join(args.output_dir, "checkpoints"))
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, "checkpoints", removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    try:
                        if not use_deepspeed:
                            if accelerator.is_main_process:
                                save_path = os.path.join(args.output_dir, 'checkpoints', f"checkpoint-{global_step}")
                                accelerator.save_state(save_path)
                                logger.info(f"Saved state to {save_path}")
                        else:
                            save_path = os.path.join(args.output_dir, 'checkpoints', f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")
                    except Exception as e:
                        logger.error(f"Error saving state: {e}")

                    if args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_transformer3d.store(transformer3d.parameters())
                        ema_transformer3d.copy_to(transformer3d.parameters())
                    log_validation(
                        args, config, vggt, geo_adapter, wan_vae, text_encoder, tokenizer, transformer3d, clip_image_encoder, 
                        accelerator, weight_dtype, global_step, dataset_names=dataset_names, dataset_weights=dataset_weights
                    )
                    if args.use_ema:
                        # Switch back to the original transformer3d parameters.
                        ema_transformer3d.restore(transformer3d.parameters())

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        
        # save and validate at the end of every epoch
        try:
            if not use_deepspeed:
                if accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, 'checkpoints', f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
            else:
                save_path = os.path.join(args.output_dir, 'checkpoints', f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
        
        if args.use_ema:
            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
            ema_transformer3d.store(transformer3d.parameters())
            ema_transformer3d.copy_to(transformer3d.parameters())
        log_validation(
            args, config, vggt, geo_adapter, wan_vae, text_encoder, tokenizer, transformer3d, clip_image_encoder, 
            accelerator, weight_dtype, global_step, dataset_names=dataset_names, dataset_weights=dataset_weights
        )
        if args.use_ema:
            # Switch back to the original transformer3d parameters.
            ema_transformer3d.restore(transformer3d.parameters())

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # transformer3d = unwrap_model(transformer3d)
        # if args.use_ema:
        #     ema_transformer3d.copy_to(transformer3d.parameters())
        try:
            save_path = os.path.join(args.output_dir, 'checkpoints', f"checkpoint-last")
            accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
        
        transformer3d = WanTransformer3DModel.from_pretrained(os.path.join(save_path, "transformer"))
        pipeline = Gen3RPipeline(
            vggt=accelerator.unwrap_model(vggt).to(weight_dtype), 
            geo_adapter=accelerator.unwrap_model(geo_adapter).to(weight_dtype),
            wan_vae=accelerator.unwrap_model(wan_vae).to(weight_dtype),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            transformer=transformer3d,
            scheduler=noise_scheduler,
            clip_image_encoder=clip_image_encoder,
        )
        pipeline.save_pretrained(os.path.join(args.output_dir, "pipelines", 'pipeline-last'))
        logger.info(f"Saved pipeline to {os.path.join(args.output_dir, 'pipelines', 'pipeline-last')}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
