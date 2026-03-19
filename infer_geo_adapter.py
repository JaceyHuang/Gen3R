import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import torch
import math
from datetime import datetime
import argparse
from accelerate.utils import set_seed
import imageio
import json
from einops import rearrange
from torchvision.transforms.functional import resize

from gen3r.models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from gen3r.models.vggt.utils.geometry import unproject_depth_map_to_point_map
from gen3r.utils.data_utils import center_crop
from gen3r.utils.common_utils import colorize_depth_map, downsample_and_save_pointcloud, save_videos_grid
from gen3r.pipeline import Gen3RPipeline

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['NCCL_DEBUG'] = 'ERROR'


def save_results(args, results):
    for key, value in results.items():
        
        if key == "rgbs" and value is not None:  
            rgb = rearrange(value, "b f h w c -> b c f h w").float().cpu()
            filename = os.path.join(args.output_dir, f"rgb.mp4")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            save_videos_grid(rgb, filename, rescale=False)

        # elif key == "depth_maps":
        #     # colorize the depth map
        #     depth = colorize_depth_map(value[0])[None, ...]  # [F, H, W, 1] -> [B, F, H, W, 3]
        #     depth = rearrange(depth, "b f h w c -> b c f h w").float().cpu()
        #     filename = os.path.join(args.output_dir, f"depth.mp4")
        #     os.makedirs(os.path.dirname(filename), exist_ok=True)
        #     save_videos_grid(depth, filename, rescale=False)

        elif key == "pcds":
            downsample_and_save_pointcloud(
                os.path.join(args.output_dir, f"pcds.ply"),
                value[0],
                results['rgbs'][0],
                voxel_size=0.005 if not args.remove_far_points else 0.003,
                filter_outliers=True, 
                depth_percentile=0,
                remove_far_points=args.remove_far_points,
            )
            
        elif key == 'cameras':
            filename = os.path.join(args.output_dir, f"cameras.json")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            extrinsics, Ks = value  # [B, F, 3, 4], [B, F, 3, 3]
            extrinsics = torch.cat(
                [extrinsics, 
                 torch.tensor([0, 0, 0, 1], device=extrinsics.device).view(1, 1, 1, 4).repeat(extrinsics.shape[0], extrinsics.shape[1], 1, 1)], dim=2)  # [B, F, 4, 4]
            
            cameras = {
                'extrinsics': extrinsics[0].float().cpu().numpy().tolist(),
                'intrinsics': Ks[0].float().cpu().numpy().tolist(),
            }
            with open(filename, "w") as f:
                json.dump(cameras, f, indent=4)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--pretrained_model_name_or_path", 
        type=str, 
        default='./checkpoints'
    )
    args.add_argument(
        "--frame_path", 
        nargs='+', 
        required=True,
        help=(
            "Path to the conditional images or video. "
            "For the allview task, this can be either a directory containing all frames "
            "or a path to the conditional video. "
            "For the other tasks, this should be the path to the conditional image(s)."
        )
    )
    args.add_argument(
        "--output_dir", 
        type=str, 
        default='./results'
    )
    args.add_argument(
        "--remove_far_points", 
        action='store_true', 
        help="Whether to remove far points in the point cloud."
    )
    args = args.parse_args()

    args.output_dir = os.path.join(args.output_dir, f'{datetime.now().strftime("%Y-%m-%d-%H-%M-00")}')
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = Gen3RPipeline.from_pretrained(args.pretrained_model_name_or_path)
    pipeline.to(device).to(torch.bfloat16)

    # load control images
    control_images = torch.from_numpy(imageio.v3.imread(args.frame_path[0]))  # [num_frames, H, W, 3]
    control_images = control_images[:49, ...]  # [num_frames, 3, H, W]
    
    control_images = control_images.permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0  # [1, num_frames, 3, H, W]
    frame_height, frame_width = control_images.shape[3], control_images.shape[4]
    scale = 560 / min(frame_height, frame_width)
    new_height, new_width = round(frame_height * scale), round(frame_width * scale)
    control_images = resize(control_images[0], [new_height, new_width])
    control_images = center_crop(control_images, (560, 560))[None, ...].to(device, torch.bfloat16)  # [1, num_frames, 3, 560, 560]
    
    # vggt
    with torch.no_grad():
        aggregated_token_list, ps_idx = pipeline.vggt.aggregator(control_images)
        h = int(math.sqrt(aggregated_token_list[0].shape[-2]-ps_idx))  # 40
        w = int(math.sqrt(aggregated_token_list[0].shape[-2]-ps_idx))  # 40
        
        # save camera and register tokens, [B, F, 1, D]
        cam_tokens = aggregated_token_list[-1][:, :, :1]
        # broadcast the camera tokens to [B, F, h, w, D]
        cam_tokens = cam_tokens.unsqueeze(2).repeat(1, 1, h, w, 1)
        
        # convert to 4 * [B, F, 40, 40, C] and ensure contiguous layout
        vggt_tokens_list = [
            rearrange(aggregated_token_list[i][:, :, ps_idx:], 'b f (h w) c -> b f h w c', h=h, w=w).contiguous()
            for i in pipeline.vggt.depth_head.intermediate_layer_idx
        ] 
        vggt_tokens_list.append(cam_tokens)  # 5 * [B, F, 40, 40, C]
        # turn the tokens into size [B, F, 40, 40, 5C]
        vggt_tokens = rearrange(torch.cat(vggt_tokens_list, dim=-1), 'b f h w c -> b c f h w')  # [B, 10240, F, 40, 40]
        
        # encode the tokens
        encoded_vggt_tokens = pipeline.geo_adapter.encode(vggt_tokens).latent_dist.sample()  # [B, 16, f, 70, 70]

        # decode
        decoded_vggt_tokens = pipeline.geo_adapter.decode(encoded_vggt_tokens).sample
        decoded_vggt_tokens = rearrange(decoded_vggt_tokens, 'b c f h w -> b f h w c')

        B, T, H, W, D = decoded_vggt_tokens.shape
        D = D // 5
        patch_size = pipeline.vggt.aggregator.patch_size

        fake_frames = torch.zeros(
            (B, T, 3, H*patch_size, W*patch_size), device=decoded_vggt_tokens.device, dtype=decoded_vggt_tokens.dtype)
        cam_register_tokens = decoded_vggt_tokens[..., -D:].mean(dim=(2, 3)).unsqueeze(2).repeat(1, 1, 5, 1)  # [B, T, 5, 2048]
        aggregated_token_list = [
            torch.cat([cam_register_tokens, decoded_vggt_tokens[..., :D].reshape(B, T, -1, D)], dim=2),
            torch.cat([cam_register_tokens, decoded_vggt_tokens[..., D:2*D].reshape(B, T, -1, D)], dim=2),
            torch.cat([cam_register_tokens, decoded_vggt_tokens[..., 2*D:3*D].reshape(B, T, -1, D)], dim=2),
            torch.cat([cam_register_tokens, decoded_vggt_tokens[..., 3*D:4*D].reshape(B, T, -1, D)], dim=2),
        ]

        pose_enc = pipeline.vggt.camera_head(aggregated_token_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, fake_frames.shape[-2:])
        depth_maps, _ = pipeline.vggt.depth_head(aggregated_token_list, fake_frames, 5)  # [B, T, H, W, 1]
        point_maps, _ = pipeline.vggt.point_head(aggregated_token_list, fake_frames, 5)  # [B, T, H, W, 3]
        point_map_by_unprojection = torch.from_numpy(unproject_depth_map_to_point_map(
            depth_maps.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0))).unsqueeze(0)  # [B, F, H, W, 3]
        
        artifacts = {
            'rgbs': rearrange(control_images, 'b f c h w -> b f h w c'),
            'depth_maps': depth_maps,
            'pcds': point_map_by_unprojection,
            'cameras': (extrinsic, intrinsic),
        }

    save_results(args, artifacts)