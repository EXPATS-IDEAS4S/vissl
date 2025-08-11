# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Any, Dict, List, Sequence
import numpy as np
import torch
import torchvision.transforms.functional as F
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform


@register_transform("ImgPilToMultiCropWithTime")
class ImgPilToMultiCropWithTime(ClassyTransform):
    """
    Multi-resolution spatial crops (using only tensor operations), optionally followed by temporal cropping.
    Works with (C, T, H, W) tensors or single-frame images.

    1. Spatial crop applied per frame (all time frames kept initially).
    2. Optional temporal crop to select overlapping time windows.
    """

    def __init__(
        self,
        total_num_crops: int,
        num_crops: Sequence[int],
        size_crops: Sequence[int],
        crop_scales: Sequence[Sequence[float]],
        temporal_crop: bool = False,
        temporal_window: int = None,
        temporal_overlap: float = 0.75,
    ):
        assert np.sum(num_crops) == total_num_crops
        assert len(size_crops) == len(num_crops)
        assert len(size_crops) == len(crop_scales)

        self.temporal_crop = temporal_crop
        self.temporal_window = temporal_window
        self.temporal_overlap = temporal_overlap

        # Prepare parameters for each crop type (repeat params for num_crops)
        self.crop_params = []
        for num, size, scale in zip(num_crops, size_crops, crop_scales):
            for _ in range(num):
                self.crop_params.append((size, scale))

    def __call__(self, video: torch.Tensor) -> List[torch.Tensor]:
        """
        video: (C, T, H, W) tensor OR (C, H, W) single frame
        """
        if video.ndim == 3:
            video = video.unsqueeze(1)  # (C, 1, H, W)

        C, T, H, W = video.shape

        crops = []
        for size, scale in self.crop_params:
            # For each crop, apply the same spatial crop params across all T frames

            # Generate parameters for RandomResizedCrop:
            # RandomResizedCrop uses scale and ratio; here we only use scale (ratio=1 for square)
            # So we need to generate random crop coordinates and resize.

            # Compute target area range
            area = H * W
            for attempt in range(10):
                target_area = random.uniform(scale[0], scale[1]) * area
                aspect_ratio = 1.0  # square crops only

                w = int(round((target_area * aspect_ratio) ** 0.5))
                h = int(round((target_area / aspect_ratio) ** 0.5))

                if w <= W and h <= H:
                    top = random.randint(0, H - h)
                    left = random.randint(0, W - w)
                    break
            else:
                # Fallback to center crop
                w = min(W, H)
                h = w
                top = (H - h) // 2
                left = (W - w) // 2
            
            print(f"Spatial crop params - top: {top}, left: {left}, height: {h}, width: {w}")

            cropped_frames = []
            for t in range(T):
                frame = video[:, t]  # (C, H, W)
                cropped = frame[:, top : top + h, left : left + w]
                resized = F.resize(cropped, [size, size])  # resize to output size
                cropped_frames.append(resized)

            crop_tensor = torch.stack(cropped_frames, dim=1)  # (C, T, size, size)
            crops.append(crop_tensor)

        # Temporal cropping (if enabled and more than 1 frame)
        if self.temporal_crop and T > 1:
            crops = self._apply_temporal_crop(crops)

        return crops

    def _apply_temporal_crop(self, crops: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply temporal cropping with required overlap between first two views.
        """
        C, T, _, _ = crops[0].shape
        win_len = self.temporal_window or T
        if win_len > T:
            win_len = T

        overlap_frames = int(win_len * self.temporal_overlap)
        shift_range = win_len - overlap_frames
        if shift_range < 0:
            shift_range = 0

        start1 = random.randint(0, T - win_len)
        start2_min = max(0, start1 - shift_range)
        start2_max = min(T - win_len, start1 + shift_range)
        start2 = random.randint(start2_min, start2_max)

        print(f"Temporal crop starts - start1: {start1}, start2: {start2}")
        
        cropped_temporal = []
        for i, view in enumerate(crops):
            if i == 0:
                start = start1
            elif i == 1:
                start = start2
            else:
                # For extra crops: just match view 1's time window
                start = start1
            cropped_temporal.append(view[:, start : start + win_len])

        return cropped_temporal

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPilToMultiCropWithTime":
        return cls(**config)

