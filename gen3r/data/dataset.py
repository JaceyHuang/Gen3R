import random
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override

from ..utils.data_utils import (
    load_prompts,
    load_extracted_videos,
    load_cameras,
    preprocess_extracted_video_with_resize_crop,
    preprocess_cameras,
    clean_prompt,
)


class BaseDataset(Dataset):
    """
    Base dataset class for geometry-aware video training.

    The dataset reads three metadata files (camera paths, prompt-file paths,
    and extracted-video directories), validates that camera/video entries are
    aligned, and builds one training sample per index.

    For each sample:
      - one prompt line is randomly selected from the prompt file,
      - camera intrinsics/extrinsics are loaded and preprocessed,
      - frames are loaded from the extracted video directory and transformed,
      - camera parameters are adjusted when invalid frames are skipped/padded.

    Subclasses must implement preprocessing hooks for frames, cameras, and
    post-load video transforms.

    Args:
        data_root (str): Root directory containing dataset metadata files.
        camera_column (str): Relative path to the file listing camera files.
        caption_column (str): Relative path to the file listing prompt files.
        video_column (str): Relative path to the file listing extracted-video directories.

    Returns:
        Dict[str, Any] from ``__getitem__`` with keys:
            - ``text``: selected prompt string
            - ``pixel_values``: frame tensor ``[F, C, H, W]``
            - ``c2ws``: camera-to-world poses
            - ``Ks``: camera intrinsics
    """

    def __init__(
        self,
        data_root: str,
        camera_column: str,
        caption_column: str,
        video_column: str,
    ) -> None:
        super().__init__()

        data_root = Path(data_root)
        self.cameras = load_cameras(data_root / camera_column)
        self.prompts = load_prompts(data_root / caption_column)
        self.videos = load_extracted_videos(data_root / video_column)

        # Check if number of cameras matches number of videos
        if len(self.videos) != len(self.cameras):
            raise ValueError(
                f"Expected length of cameras and videos to be the same but found {len(self.cameras)=} and {len(self.videos)=}. Please ensure that the number of cameras and videos match in your dataset."
            )


    def __len__(self) -> int:
        return len(self.videos)


    def __getitem__(self, index: int) -> Dict[str, Any]:

        prompt_path = self.prompts[index]  # the caption file path, there are 2 prompts in the file, so we randomly select one
        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = file.readlines()
            prompts = clean_prompt(random.choice(prompts))

        camera_path = self.cameras[index]
        # get camera intrinsics and poses
        Ks, c2ws, f_names = self.preprocess_cameras(camera_path, interval=np.random.randint(1, self.max_interval))

        video_dir = self.videos[index]  # the images directory of the extracted video
        frames, invalid_indices, num_first_repeats, num_last_repeats = self.preprocess(video_dir, f_names)
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
        
        # Current shape of frames: [F, C, H, W]
        frames = self.video_transform(frames).clamp(0, 1)

        sample = {
            "Ks": Ks,
            "c2ws": c2ws,
            "text": prompts,
            "pixel_values": frames,
        }
        
        return sample


    def preprocess(self, video_path: Path) -> torch.Tensor:
        """
        Loads and preprocesses a video.

        Args:
            video_path: Path to the video file to load.

        Returns:
            torch.Tensor: Video tensor of shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width
        """
        raise NotImplementedError("Subclass must implement this method")


    def preprocess_cameras(self, camera_path: Path) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Loads and preprocesses a camera path.
        """
        raise NotImplementedError("Subclass must implement this method")


    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to a video.

        Args:
            frames (torch.Tensor): A 4D tensor representing a video
                with shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed video tensor with the same shape as the input
        """
        raise NotImplementedError("Subclass must implement this method")


class DatasetWithResizeCrop(BaseDataset):
    """
    A dataset class for text-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos
        width (int): Target width for resizing videos
    """

    def __init__(self, max_num_frames: int, height: int, width: int, max_interval: int=None, fix_start_frame: int=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width
        self.max_interval = max_interval
        self.fix_start_frame = fix_start_frame
        self.__frame_transform = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 )]  # only normalize to 0~1
        )

    @override
    def preprocess(self, video_path: Path, f_names: List[str] | None = None) -> torch.Tensor:
        return preprocess_extracted_video_with_resize_crop(
            video_path,
            self.height,
            self.width,
            f_names,
        )

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transform(f) for f in frames], dim=0)
    
    @override
    def preprocess_cameras(self, camera_path: Path, interval: int) -> torch.Tensor:
        return preprocess_cameras(
            camera_path,
            self.max_num_frames,
            self.height,
            self.width,
            interval,
            self.fix_start_frame,
        )


class BalancedDatasetWithResizeCrop(Dataset):
    """
    A dataset class that provides balanced sampling across multiple datasets.
    
    This class creates multiple SGDatasetWithResizeCrop instances for different datasets
    and samples from them with equal probability, regardless of their individual sizes.
    
    Args:
        dataset_configs (List[Dict]): List of dataset configurations, each containing:
            - data_root: Root directory for the dataset
            - video_column: Path to video metadata file
            - camera_column: Path to camera metadata file  
            - caption_column: Path to caption metadata file
            - dataset_name: Name identifier for the dataset (used for logging)
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos
        width (int): Target width for resizing videos
        interval (int, optional): Frame sampling interval
        max_interval (int, optional): Maximum frame sampling interval
        fix_start_frame (bool, optional): Whether to fix the starting frame
        dataset_weights (List[float], optional): Weights for each dataset. If None, equal weights are used.
    """
    
    def __init__(
        self,
        dataset_configs: List[Dict[str, Any]],
        max_num_frames: int,
        height: int,
        width: int,
        max_interval: int = None,
        fix_start_frame: int = None,
        dataset_weights: List[float] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.dataset_configs = dataset_configs
        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width
        self.max_interval = max_interval
        self.fix_start_frame = fix_start_frame
        
        # Create individual datasets
        self.datasets = []
        self.dataset_names = []
        self.dataset_sizes = []
        
        for config in dataset_configs:
            dataset = DatasetWithResizeCrop(
                data_root=config['data_root'],
                video_column=config['video_column'],
                camera_column=config['camera_column'],
                caption_column=config['caption_column'],
                max_num_frames=max_num_frames,
                height=height,
                width=width,
                max_interval=max_interval,
                fix_start_frame=fix_start_frame,
                *args,
                **kwargs,
            )
            self.datasets.append(dataset)
            self.dataset_names.append(config.get('dataset_name', f'dataset_{len(self.datasets)}'))
            self.dataset_sizes.append(len(dataset))
        
        # Set up sampling weights
        if dataset_weights is None:
            # Equal weights for all datasets
            self.dataset_weights = [1.0] * len(self.datasets)
        else:
            if len(dataset_weights) != len(self.datasets):
                raise ValueError(f"Number of dataset weights ({len(dataset_weights)}) must match number of datasets ({len(self.datasets)})")
            self.dataset_weights = dataset_weights
        
        # Normalize weights to probabilities
        total_weight = sum(self.dataset_weights)
        self.dataset_probs = [w / total_weight for w in self.dataset_weights]
        
        # Calculate total length (sum of all dataset sizes)
        self.total_length = sum(self.dataset_sizes)
        
        # Print dataset statistics
        print(f"BalancedSGDatasetWithResizeCrop initialized with {len(self.datasets)} datasets:")
        for i, (name, size, prob) in enumerate(zip(self.dataset_names, self.dataset_sizes, self.dataset_probs)):
            print(f"  {name}: {size} samples, probability: {prob:.3f}")
        print(f"Total samples: {self.total_length}\n")
    

    def __len__(self) -> int:
        return self.total_length
    

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Sample dataset based on probabilities
        dataset_idx = np.random.choice(len(self.datasets), p=self.dataset_probs)
        
        # Sample from the selected dataset
        dataset = self.datasets[dataset_idx]
        sample_idx = np.random.randint(0, len(dataset))
        
        # Get sample and add dataset metadata
        sample = dataset[sample_idx]
        sample['dataset_name'] = self.dataset_names[dataset_idx]
        sample['dataset_idx'] = dataset_idx
        
        return sample
    
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Return statistics about the datasets and sampling probabilities."""
        return {
            'dataset_names': self.dataset_names,
            'dataset_sizes': self.dataset_sizes,
            'dataset_weights': self.dataset_weights,
            'dataset_probs': self.dataset_probs,
            'total_length': self.total_length,
        }