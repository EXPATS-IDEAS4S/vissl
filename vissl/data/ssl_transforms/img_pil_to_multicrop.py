# ImgPilToMultiCropWithTime takes a video tensor shaped (C, T, H, W) 
# and produces multiple spatial crops (applied identically to every time frame)
#  and optionally temporal sub-windows of those crops. 
# Spatial cropping mimics RandomResizedCrop (area-based random crop + resize) 
# but implemented on tensors. Temporal cropping chooses overlapping time windows 
# for the first two views.

# Readapted from ImgPilToMultiCrop by Daniele Corradini

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
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
        spatial_crop: bool = True,
        aspect_ratios: Sequence[Sequence[float]] = [[1.0, 1.0]],
        temporal_crop: bool = False,
        temporal_window: int = None,
        temporal_overlap_min: float = 0.10,
        temporal_overlap_max: float = 1.0,
    ):
        assert np.sum(num_crops) == total_num_crops
        assert len(size_crops) == len(num_crops)
        assert len(size_crops) == len(crop_scales)
        assert len(size_crops) == len(aspect_ratios)

        self.temporal_crop = temporal_crop
        self.temporal_window = temporal_window
        self.temporal_overlap_min = temporal_overlap_min
        self.temporal_overlap_max = temporal_overlap_max
        self.spatial_crop = spatial_crop

        # Prepare parameters for each crop type (repeat params for num_crops)
        self.crop_params = []
        for num, size, scale, aspect_ratio in zip(num_crops, size_crops, crop_scales, aspect_ratios):
            for _ in range(num):
                self.crop_params.append((size, scale, aspect_ratio))

    def __call__(self, video: torch.Tensor) -> List[torch.Tensor]:
        """
        video: (C, T, H, W) tensor OR (C, H, W) single frame
        """
        if video.ndim == 3:
            video = video.unsqueeze(1)  # (C, 1, H, W)

        C, T, H, W = video.shape

        crops = []
        crops_info = []  # to store crop parameters for plotting

        for size, scale, aspect_ratio in self.crop_params:
            if self.spatial_crop:
                area = H * W
                for attempt in range(10):
                    target_area = random.uniform(scale[0], scale[1]) * area
                    random_aspect_ratio = random.uniform(aspect_ratio[0], aspect_ratio[1])
                    w = int(round((target_area * random_aspect_ratio) ** 0.5))
                    h = int(round((target_area / random_aspect_ratio) ** 0.5))

                    if w <= W and h <= H:
                        top = random.randint(0, H - h)
                        left = random.randint(0, W - w)
                        break
                else:
                    w = min(W, H)
                    h = w
                    top = (H - h) // 2
                    left = (W - w) // 2
                print(f"Cropping frame: top={top}, left={left}, height={h}, width={w}")

                cropped_frames = []
                for t in range(T):
                    frame = video[:, t]  # (C, H, W)
                    cropped = frame[:, top : top + h, left : left + w]
                    resized = F.resize(cropped, [size, size])
                    cropped_frames.append(resized)

                crop_tensor = torch.stack(cropped_frames, dim=1)  # (C, T, size, size)
                crops.append(crop_tensor)

                # Store crop info - before temporal cropping
                crops_info.append({
                    'top': top,
                    'left': left,
                    'height': h,
                    'width': w,
                    'temporal_start': 0,
                    'temporal_len': T
                })
            else:
                # If no spatial cropping, just keep the original video
                crops.append(video)
                crops_info.append({
                    'top': 0,
                    'left': 0,
                    'height': H,
                    'width': W,
                    'temporal_start': 0,
                    'temporal_len': T
                })

        # Apply temporal cropping if enabled
        if self.temporal_crop and T > 1:
            crops, starts = self._apply_temporal_crop(crops)

            # Update temporal info based on temporal cropping
            # (Assuming _apply_temporal_crop returns cropped tensors with length <= T)
            for i, crop in enumerate(crops):
                crops_info[i]['temporal_start'] = starts[i]  # new temporal length
                crops_info[i]['temporal_len'] = self.temporal_window # new temporal length

            # If you know the temporal start indices, update 'temporal_start' in crops_info accordingly
            # For example, if your _apply_temporal_crop method returns the starts, set them here.

        # ----- PLOTTING -----
        num_views = len(crops)
        fig, axs = plt.subplots(num_views, T, figsize=(3 * T, 3 * num_views))

        for v in range(num_views):
            crop = crops[v]  # (C, crop_T, crop_H, crop_W)
            info = crops_info[v]
            start = info['temporal_start']
            end = start + info['temporal_len']
            top = info['top']
            left = info['left']
            width = info['width']
            height = info['height']

            for t in range(T):
                ax = axs[v, t] if num_views > 1 else axs[t]

                # Always plot the original full frame first
                full_frame = video[:, t].cpu().numpy()
                if C == 1:
                    ax.imshow(full_frame.squeeze(), cmap='gray_r', vmin=200, vmax=300)
                else:
                    ax.imshow(full_frame[0], cmap='gray_r', vmin=200, vmax=300)  # first channel only

                # If t is inside crop's temporal window, overlay the spatial region
                if start <= t < end:
                    if C == 1:
                        crop_region = full_frame[0, top:top+height, left:left+width]
                    else:
                        crop_region = full_frame[0, top:top+height, left:left+width]

                    # Create a masked overlay so only that region is colored
                    overlay = np.full_like(full_frame[0], np.nan, dtype=float)  # NaNs mean transparent in imshow
                    overlay[top:top+height, left:left+width] = crop_region

                    ax.imshow(overlay, cmap='Reds', vmin=200, vmax=300, alpha=0.5)

                ax.set_title(f"View {v} Time {t}")
                ax.axis('off')

        fig.savefig("/data1/runs/dcv2_ir108_100x100_k9_1k_nc_r2dplus1/crop_time_all.png", bbox_inches='tight')
        plt.close()

        # --------------------

        return crops

    def _apply_temporal_crop(self, crops: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply temporal cropping with required min/max overlap between first two views.
        """
        C, T, _, _ = crops[0].shape
        win_len = self.temporal_window or T
        win_len = min(win_len, T)  # clamp if longer than available frames

        # Convert overlap ratios to frames
        min_overlap_frames = int(win_len * self.temporal_overlap_min)
        max_overlap_frames = int(win_len * self.temporal_overlap_max)

        # Ensure they are within valid bounds
        min_overlap_frames = max(0, min(min_overlap_frames, win_len))
        max_overlap_frames = max(min_overlap_frames, min(max_overlap_frames, win_len))

        # Random start for first view
        start1 = random.randint(0, T - win_len)

        # Determine allowed range for start2 based on min/max overlap
        start2_min = max(0, start1 - (win_len - max_overlap_frames))
        start2_max = min(T - win_len, start1 + (win_len - min_overlap_frames))

        # Pick start2 within that range
        start2 = random.randint(start2_min, start2_max)

        print(f"Temporal crop starts - start1: {start1}, start2: {start2}")
        print(f"Overlap frames range: {min_overlap_frames}–{max_overlap_frames}")

        cropped_temporal = []
        starts = []
        for i, view in enumerate(crops):
            if i == 0:
                start = start1
            elif i == 1:
                start = start2
            else:
                start = start1  # match view 1 for others

            starts.append(start)
            cropped_temporal.append(view[:, start:start + win_len])

        return cropped_temporal, starts


    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPilToMultiCropWithTime":
        return cls(**config)

