import os
import re
import imageio
import random
import json
import torch
import numpy as np

from einops import rearrange
from pathlib import Path
from typing import List, Tuple
from torchvision.transforms.functional import resize, InterpolationMode


def load_cameras(camera_path: Path) -> List[Path]:
    with open(camera_path, "r", encoding="utf-8") as file:
        return [Path(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
    

def load_prompts(prompt_path: Path) -> List[str]:
    with open(prompt_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if len(line.strip()) > 0]


def load_extracted_videos(video_path: Path) -> List[Path]:
    with open(video_path, "r", encoding="utf-8") as file:
        return [Path(line.strip()) for line in file.readlines() if len(line.strip()) > 0]


def create_dataset_config(args):
    """
    Helper function to create dataset configs.
    """
    
    # This assumes your current data structure has dataset names in the paths
    # Adjust the logic based on your actual data organization
    
    dataset_configs = []
    
    # List of your dataset names
    dataset_info = {
        'dl3dv': 1.0,
        'acid': 1.0,
        'mvimgnet': 0.25,
        'waymo': 0.5,
        'co3d': 1.0,
        're10k': 1.0,
        'kitti360': 0.5,
        'vkitti': 0.5,
        'wildrgbd': 1.0,
        'tartanair': 1.0,
    }

    dataset_names = dataset_info.keys()
    dataset_weights = dataset_info.values()
    
    for dataset_name in dataset_names:
        data_root = f"{args.train_data_dir}/{dataset_name}"
        config = {
            'data_root': data_root,
            'video_column': 'train_videos_dirs.txt',
            'camera_column': 'train_cameras_paths.txt',
            'caption_column': 'train_captions_paths.txt',
            'dataset_name': dataset_name
        }
        dataset_configs.append(config)

    return dataset_configs, dataset_names, dataset_weights


def _get_dataset_type_from_path(path: Path, dataset_names: List[str]) -> str:
    """
    Determine dataset type from file path.
    
    Args:
        path: Path to the file
        dataset_names: List of dataset names to check against
    Returns:
        Dataset type string
    """
    path_str = str(path).lower()
    for dataset_name in dataset_names:
        if dataset_name in path_str:
            return dataset_name
    return 'unknown'


def clean_prompt(prompt: str) -> str:
    """
    Clean the prompt to remove fixed prefixes.
    """
    # a too long prompt is probably bad
    if len(prompt) > 1000:
        return ''
    
    if random.random() < 0.2:
        return prompt

    prefixes = [
        r"^The image (?:shows|portrays|depicts|showcases|presents|captures|features|displays|comprises|includes) (?P<art>a|an)(?=\s+(?P<next>\w+))",
        r"^The images (?:show|portray|depict|showcase|present|capture|feature|display|comprise|include) (?P<art>a|an)(?=\s+(?P<next>\w+))",
        r"^The (?:scene|scene in the images|scene captured in the images|scene depicted in the images|main visual content of the scene) (?:is|is of|shows|portrays|depicts|showcases|presents|captures|features|displays|comprises|includes) (?P<art>a|an)(?=\s+(?P<next>\w+))",
    ]
    patterns = [re.compile(p) for p in prefixes]

    def choose_article(word):
        return "an" if word and word[0].lower() in "aeiou" else "a"

    def repl(m):
        art = choose_article(m.group("next"))
        if random.random() < 0.75:
            return "An" if art == "an" else "A"
        else:
            return "It is an" if art == "an" else "It is a"

    prompt = prompt.strip()
    for pat in patterns:
        new_prompt, n = pat.subn(repl, prompt, count=1)
        if n:
            prompt = new_prompt
            break
    
    return prompt


def select_prompts(prompt_paths: List[Path], dataset_names: List[str], dataset_weights: List[float] = None, num_selections: int = 6) -> List[Path]:
    """
    Select prompts for validation with one fixed sample and the rest randomly sampled.

    Selection strategy:
      1) Always select one prompt from the first file in `prompt_paths` (index 0).
      2) Select `num_selections - 1` prompts from the remaining files.
         - If `dataset_weights` is provided, files are sampled by dataset-level
           weighted probabilities inferred from file paths.
         - Otherwise, files are sampled uniformly.

    Empty files are skipped/replaced whenever possible. The function returns both
    the cleaned prompt strings and the original indices of the selected files.

    Args:
        prompt_paths: List of text-file paths where each file contains candidate prompts.
        dataset_names: Dataset name keywords used to map each path to a dataset type.
        dataset_weights: Optional sampling weights aligned with `dataset_names`.
            If `None`, uniform random sampling is used.
        num_selections: Total number of prompts to return.

    Returns:
        Tuple[List[str], List[int]]:
            - selected_prompts: Chosen prompt strings after `clean_prompt`.
            - selected_indices: Indices in `prompt_paths` corresponding to each prompt.
    """
    selected_prompts = []
    selected_paths = []
    selected_indices = []
    # First handle fixed prompts (first 1)
    for i, prompt_path in enumerate(prompt_paths[:1]):
        while True:
            try:
                with open(prompt_path, "r", encoding="utf-8") as file:
                    prompts = file.readlines()
                    index = i
                    if len(prompts) == 0:
                        # If the file is empty, try to find a non-empty prompt file
                        for j, candidate_path in enumerate(prompt_paths[1:]):
                            with open(candidate_path, "r", encoding="utf-8") as candidate_file:
                                if len(candidate_file.readlines()) > 0:
                                    prompt_path = candidate_path
                                    index = j + 1
                                    break
                        else:
                            raise ValueError(f"Could not find a non-empty prompt file to replace {prompt_path}")
                        continue
                    selected_prompts.append(clean_prompt(random.choice(prompts)))
                    selected_indices.append(index)
                    selected_paths.append(prompt_path)
                    break
            except Exception as e:
                raise ValueError(f"Error processing prompt file {prompt_path}: {str(e)}")

    # Then handle random prompts (num_selections-1 from remaining)
    remaining_paths = [p for p in prompt_paths[1:] if p not in selected_paths]
    if len(remaining_paths) < num_selections-1:
        raise ValueError("Not enough non-empty prompt files available for random sampling")
    
    # Set up dataset probability sampling if provided
    if dataset_weights is not None:
        # Create dataset type to probability mapping
        if len(dataset_weights) != len(dataset_names):
            raise ValueError(f"dataset_weights length ({len(dataset_weights)}) must match number of dataset types ({len(dataset_names)})")
        
        dataset_prob_map = dict(zip(dataset_names, dataset_weights))
        
        # Group remaining paths by dataset type
        paths_by_dataset = {}
        for path in remaining_paths:
            dataset_type = _get_dataset_type_from_path(path, dataset_names)
            if dataset_type not in paths_by_dataset:
                paths_by_dataset[dataset_type] = []
            paths_by_dataset[dataset_type].append(path)
        
        # Create probability distribution for sampling
        available_datasets = [dt for dt in dataset_names if dt in paths_by_dataset and len(paths_by_dataset[dt]) > 0]
        if not available_datasets:
            raise ValueError("No available datasets found in remaining paths")
        
        # Normalize probabilities for available datasets
        available_probs = [dataset_prob_map[dt] for dt in available_datasets]
        total_prob = sum(available_probs)
        normalized_probs = [p / total_prob for p in available_probs]
    else:
        # Use uniform random sampling
        available_datasets = None
        normalized_probs = None
        paths_by_dataset = None
    
    for _ in range(num_selections-1):
        while True:
            try:
                if dataset_weights is not None and available_datasets:
                    # Sample dataset type based on probabilities
                    dataset_type = np.random.choice(available_datasets, p=normalized_probs)
                    # Sample from the selected dataset type
                    prompt_path = random.choice(paths_by_dataset[dataset_type])
                else:
                    # Use uniform random sampling
                    prompt_path = random.choice(remaining_paths)
                
                with open(prompt_path, "r", encoding="utf-8") as file:
                    prompts = file.readlines()
                    index = prompt_paths.index(prompt_path)
                    if paths_by_dataset is not None:
                        paths_by_dataset[dataset_type].remove(prompt_path)
                    remaining_paths.remove(prompt_path)
                    
                    if len(prompts) == 0:
                        # If the file is empty, try to find a non-empty prompt file
                        if dataset_weights is not None and available_datasets:
                            # Try to find another file from the same dataset type
                            for candidate_path in paths_by_dataset.get(dataset_type, []):
                                if candidate_path != prompt_path and candidate_path in remaining_paths:
                                    with open(candidate_path, "r", encoding="utf-8") as candidate_file:
                                        prompts = candidate_file.readlines()
                                        if len(prompts) > 0:
                                            prompt_path = candidate_path
                                            index = prompt_paths.index(prompt_path)
                                            paths_by_dataset[dataset_type].remove(prompt_path)
                                            remaining_paths.remove(prompt_path)
                                            break
                            # Fall back to any non-empty file
                            while True:
                                new_prompt_path = random.choice(remaining_paths)
                                if new_prompt_path != prompt_path:
                                    with open(new_prompt_path, "r", encoding="utf-8") as candidate_file:
                                        prompts = candidate_file.readlines()
                                        if len(prompts) > 0:
                                            prompt_path = candidate_path
                                            index = prompt_paths.index(prompt_path)
                                            remaining_paths.remove(prompt_path)
                                            break
                        else:
                            while True:
                                new_prompt_path = random.choice(remaining_paths)
                                if new_prompt_path != prompt_path:
                                    with open(new_prompt_path, "r", encoding="utf-8") as candidate_file:
                                        prompts = candidate_file.readlines()
                                        if len(prompts) > 0:
                                            prompt_path = new_prompt_path
                                            index = prompt_paths.index(prompt_path)
                                            remaining_paths.remove(prompt_path)
                                            break
                    selected_prompts.append(clean_prompt(random.choice(prompts)))
                    selected_indices.append(index)
                    selected_paths.append(prompt_path)
                    break
            except Exception as e:
                raise ValueError(f"Error processing prompt file {prompt_path}: {str(e)}")

    assert len(selected_prompts) == len(selected_indices) == num_selections, f"Expected {num_selections} prompts, but got {len(selected_prompts)}"
    return selected_prompts, selected_indices


def preprocess_cameras(
    camera_path: Path,
    max_num_frames: int,
    height: int,
    width: int,
    interval: int = 2,
    fix_start_frame: int | None = None,
    repeat_last: bool = False,
    total_frames: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    try:
        with open(camera_path, "r", encoding="utf-8") as file:
            transforms = json.load(file)
    except Exception as e:
        raise ValueError(f"Error processing camera file {camera_path}: {str(e)}")
    
    c2ws, frame_paths, Ks = [], [], []
    for frame in transforms["frames"]:
        w, h, fl_x, fl_y, cx, cy = frame["w"], frame["h"], frame["fl_x"], frame["fl_y"], frame["cx"], frame["cy"]
        K = get_K(w, h, fl_x, fl_y, cx, cy, width, height)
        c2w = torch.tensor(frame["transform_matrix"])  # in OPENCV convention
        Ks.append(K)
        c2ws.append(c2w)
        frame_paths.append(frame["file_path"])
    Ks = torch.stack(Ks, dim=0)
    c2ws = torch.stack(c2ws, dim=0)

    assert len(c2ws) > 0, f"No valid frames found in file {camera_path}"

    if total_frames > 0:
        Ks = Ks[:total_frames]
        c2ws = c2ws[:total_frames]
        frame_paths = frame_paths[:total_frames]
    
    if len(c2ws) < max_num_frames * interval:
        interval = 1  # try to avoid repeating the same frame

    if len(c2ws) < max_num_frames * interval:
        num_repeats = max_num_frames * interval - len(c2ws)
        if not repeat_last:
            num_first_repeats = random.randint(0, num_repeats)
        else:
            num_first_repeats = 0
        num_last_repeats = num_repeats - num_first_repeats

        if num_first_repeats > 0:
            first_K = Ks[:1]
            first_c2w = c2ws[:1]
            first_frame_path = frame_paths[0]
            repeated_Ks = first_K.repeat(num_first_repeats, 1, 1)
            repeated_c2ws = first_c2w.repeat(num_first_repeats, 1, 1)
            repeated_frame_paths = [first_frame_path for _ in range(num_first_repeats)]
            Ks = torch.cat([repeated_Ks, Ks], dim=0)
            c2ws = torch.cat([repeated_c2ws, c2ws], dim=0)
            frame_paths = repeated_frame_paths + frame_paths
        if num_last_repeats > 0:
            last_K = Ks[-1:]
            last_c2w = c2ws[-1:]
            last_frame_path = frame_paths[-1]
            repeated_Ks = last_K.repeat(num_last_repeats, 1, 1)
            repeated_c2ws = last_c2w.repeat(num_last_repeats, 1, 1)
            repeated_frame_paths = [last_frame_path for _ in range(num_last_repeats)]
            Ks = torch.cat([Ks, repeated_Ks], dim=0)
            c2ws = torch.cat([c2ws, repeated_c2ws], dim=0)
            frame_paths = frame_paths + repeated_frame_paths
    
    if fix_start_frame is not None:
        start_index = fix_start_frame
    else:
        start_index = torch.randint(0, len(c2ws) - max_num_frames * interval + interval, (1,))
    indices = list(range(start_index, start_index + max_num_frames * interval, interval))
    
    frame_paths = [frame_paths[i].split("/")[-1] for i in indices]  # only the filename
    Ks = Ks[indices]
    c2ws = c2ws[indices]
    c2ws = preprocess_poses(c2ws)
    
    return Ks, c2ws, frame_paths


def preprocess_extracted_video_with_resize_crop(
    video_path: Path | str,
    height: int,
    width: int,
    frame_paths: List[str] | List[Path] | None = None,
    num_frames: int | None = None,
    frame_interval: int | None = None,
    fix_start_frame: int | None = None,
    repeat_last: bool = False,
) -> torch.Tensor:
    """
    Load a clip from an extracted-frame directory, then apply
    "sampling/padding + aspect-preserving resize + center crop".

    Two frame-selection modes are supported:
      1) Provide `frame_paths`: frames are read in the given order.
      2) Provide `num_frames` and `frame_interval`: frames are sampled from
         filename-sorted directory entries.

    If there are not enough frames for `num_frames * frame_interval`, the
    function first falls back to `frame_interval = 1`. If still insufficient,
    it pads by repeating the first and/or last frame. Read failures are tracked
    in `invalid_indices` and also compensated by padding to keep target length.

    Frames are stacked to `[F, 3, H, W]`, then processed as:
      - resize with the larger ratio (to fully cover target height/width)
      - center crop to `(height, width)`

    Args:
        video_path: Path to the directory containing extracted frames.
        height: Target output height.
        width: Target output width.
        frame_paths: Optional relative frame paths to load. When provided,
            this mode is used directly.
        num_frames: Target number of frames (required when `frame_paths` is
            not provided).
        frame_interval: Sampling interval (required when `frame_paths` is
            not provided).
        fix_start_frame: Fixed start index for sampling. If `None`, a random
            start index is used.
        repeat_last: If `True`, failure padding uses only the last frame;
            otherwise the padding count is randomly split between first/last.

    Returns:
        Tuple[torch.Tensor, List[int], int, int]:
            - frames: Float tensor with shape `[F, 3, height, width]`.
            - invalid_indices: Indices of frames that failed to load.
            - num_first_repeats: Number of prepended repeated frames.
            - num_last_repeats: Number of appended repeated frames.
    """
    assert (num_frames is not None and frame_interval is not None) or (frame_paths is not None), \
        "num_frames and frame_interval must be provided or frame_paths must be provided"

    if isinstance(video_path, str):
        video_path = Path(video_path)
    
    frames, invalid_indices = [], []
    if frame_paths is not None:
        for index, frame_path in enumerate(frame_paths):
            try:
                frame = imageio.imread(video_path / frame_path)  # [h, w, 3]
                if frame.shape[-1] == 4:
                    frame = frame[..., :3]
                frames.append(frame)
            except Exception as e:
                print(f"Error loading frame {str(video_path / frame_path)}: {e}")
                invalid_indices.append(index)
                continue
    else:
        # filter files with .png extension
        video_fn = sorted(os.listdir(video_path))
        video_fn = [fn for fn in video_fn if fn.endswith('.png') or fn.endswith('.jpg') or fn.endswith('.jpeg')]
        if len(video_fn) < num_frames * frame_interval:
            frame_interval = 1  # try to avoid repeating the same frame
        
        if len(video_fn) < num_frames * frame_interval:  # repeat the first or last frame
            num_repeats = num_frames * frame_interval - len(video_fn)
            num_first_repeats = random.randint(0, num_repeats)
            num_last_repeats = num_repeats - num_first_repeats
            if num_first_repeats > 0:
                first_video_fn = video_fn[0]
                repeated_video_fn = [first_video_fn] * num_first_repeats
                video_fn = repeated_video_fn + video_fn
            if num_last_repeats > 0:
                last_video_fn = video_fn[-1]
                repeated_video_fn = [last_video_fn] * num_last_repeats
                video_fn = video_fn + repeated_video_fn

        if fix_start_frame is not None:
            start_index = fix_start_frame
        else:
            start_index = torch.randint(0, len(video_fn) - num_frames * frame_interval + frame_interval, (1,))
        indices = list(range(start_index, start_index + num_frames * frame_interval, frame_interval))
        video_fn = [video_fn[i] for i in indices]

        # load frames
        for index, fn in enumerate(video_fn):
            frame_path = video_path / Path(fn)
            try:  # some images are truncated
                frame = imageio.imread(frame_path)  # [h, w, 3]
                if frame.shape[-1] == 4:
                    frame = frame[..., :3]
                frames.append(frame)
            except Exception as e:
                print(f"Error loading frame {str(video_path / frame_path)}: {e}")
                invalid_indices.append(index)
                continue
    
    if len(invalid_indices) > 0:  # repeat the last or first frame if the video is shorter than num_frames
        num_repeats = len(invalid_indices)
        if not repeat_last:
            num_first_repeats = random.randint(0, num_repeats)
        else:
            num_first_repeats = 0
        num_last_repeats = num_repeats - num_first_repeats
        if num_first_repeats > 0:
            first_frame = frames[0]
            repeated_frames = [first_frame] * num_first_repeats
            frames = repeated_frames + frames
        if num_last_repeats > 0:
            last_frame = frames[-1]
            repeated_frames = [last_frame] * num_last_repeats
            frames = frames + repeated_frames
    else:
        num_first_repeats = 0
        num_last_repeats = 0

    frames = torch.from_numpy(np.stack(frames, axis=0))
    frames = frames.float().permute(0, 3, 1, 2)  # [max_num_frames, 3, h, w]

    if num_frames is not None:
        assert len(frames) == num_frames, f"len(frames) == {len(frames)}, num_frames == {num_frames}"
    else:
        assert len(frames) == len(frame_paths), f"len(frames) == {len(frames)}, len(frame_paths) == {len(frame_paths)}"

    # height == width == 560
    img_height, img_width = frames.shape[2], frames.shape[3]
    height_resize_ratio = height / img_height
    width_resize_ratio = width / img_width
    resize_ratio = max(height_resize_ratio, width_resize_ratio)
    frames = resize(frames, (round(frames.shape[2] * resize_ratio), round(frames.shape[3] * resize_ratio)))
    frames = center_crop(frames, (height, width))

    if num_frames is not None:
        assert frames.shape == (num_frames, 3, height, width), \
            f"{str(video_path)}, frames.shape == {frames.shape}, target shape == ({num_frames}, 3, {height}, {width})"
    else:
        assert frames.shape == (len(frame_paths), 3, height, width), \
            f"{str(video_path)}, frames.shape == {frames.shape}, target shape == ({len(frame_paths)}, 3, {height}, {width})"

    return frames, invalid_indices, num_first_repeats, num_last_repeats


def center_crop(frames: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    Center crops a tensor of frames.
    """
    height, width = frames.shape[2:]
    crop_height, crop_width = size
    top = max(0, (height - crop_height) // 2)
    down = min(height, (height + crop_height) // 2)
    left = max(0, (width - crop_width) // 2)
    right = min(width, (width + crop_width) // 2)
    return frames[:, :, top:down, left:right]


def get_K(w, h, fl_x, fl_y, cx, cy, width, height):
    w_ratio, h_ratio = width / w, height / h
    resize_ratio = max(h_ratio, w_ratio)
    
    fl_x, fl_y = fl_x * resize_ratio, fl_y * resize_ratio
    cx, cy = cx * resize_ratio, cy * resize_ratio
    
    # Calculate center crop offsets
    new_h = int(h * resize_ratio)
    new_w = int(w * resize_ratio)
    crop_h = (new_h - height) // 2
    crop_w = (new_w - width) // 2
    
    # Adjust principal point for center crop
    cx -= crop_w
    cy -= crop_h
    
    # Create camera intrinsic matrix
    K = torch.tensor([
        [fl_x, 0, cx],
        [0, fl_y, cy],
        [0, 0, 1]
    ])
    return K


@torch.no_grad()
def compute_rays(c2w, K, h, w, device="cuda"):
    """
    Args:
        c2w (torch.tensor): [v, 4, 4]
        K (torch.tensor): [v, 3, 3]
        h (int): height of the image
        w (int): width of the image
    Returns:
        ray_o (torch.tensor): [v, 3, h, w]
        ray_d (torch.tensor): [v, 3, h, w]
    """

    if c2w.ndim == 4:
        c2w = c2w.squeeze(0)
    if K.ndim == 4:
        K = K.squeeze(0)

    v = c2w.shape[0]

    fx, fy, cx, cy = K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]  # [v]
    fxfycxcy = torch.stack([fx, fy, cx, cy], dim=-1).to(device)  # [v, 4]

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    x, y = x.to(device), y.to(device)
    x = x[None, :, :].expand(v, -1, -1).reshape(v, -1)
    y = y[None, :, :].expand(v, -1, -1).reshape(v, -1)
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / (fxfycxcy[:, 0:1] + 1e-8)
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / (fxfycxcy[:, 1:2] + 1e-8)
    z = torch.ones_like(x).to(device)
    ray_d = torch.stack([x, y, z], dim=2).to(c2w.dtype)  # [v, h*w, 3]
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [v, h*w, 3]
    ray_d = ray_d / (torch.norm(ray_d, dim=2, keepdim=True) + 1e-8)  # normalize in camera space
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [v, h*w, 3]

    ray_o = rearrange(ray_o, "v (h w) c -> v c h w", v=v, h=h, w=w, c=3)
    ray_d = rearrange(ray_d, "v (h w) c -> v c h w", v=v, h=h, w=w, c=3)

    return ray_o, ray_d


def preprocess_poses(
    abs_c2ws: torch.Tensor,  # [v, 4, 4]
):
    """
    Preprocess the poses to:
    1. translate and rotate the scene so that the first frame is the identity
    """
    abs_w2cs = torch.linalg.inv(abs_c2ws)
    cam_to_origin = 0
    target_cam_c2w = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).to(device=abs_w2cs.device, dtype=abs_w2cs.dtype)
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    rel_c2ws = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    rel_c2ws = torch.stack(rel_c2ws, dim=0)

    return rel_c2ws