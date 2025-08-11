# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import torchvision.transforms as pth_transforms
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
import torch
from torch import Tensor

class RandomTemporalCrop(torch.nn.Module):
    """Crop the given video at a random location in time.
    If the video is torch Tensor, it is expected
    to have [C, T, H, W] shape, where C is the number of channels, T is the number of frames,
    H is the height, and W is the width.

    Args:
        size (int): Desired output size of the crop in the dimension of time. Spatial dimentions remain untouched. 
    """

    @staticmethod
    def get_params(vid: Tensor, output_size: int) -> Tuple[int, int]:
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (Tensor): video to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        if not vid.ndim >= 3:
            raise TypeError("Tensor is not a torch video.")
        else:
            t = vid.shape[-3]
            tt = output_size

            if t < tt:
                raise ValueError(f"Required time crop size {tt} is larger than input image size {t}")

            if t == tt:
                return 0, tt
            
            i = torch.randint(0, t - tt + 1, size=(1,)).item()
            return i, tt

    def __init__(self, size):
        super().__init__()

        self.size = size
        if not isinstance(self.size, int):
            raise ValueError("Size should be a single integer representing the time dimension of the cropped video.")

    def forward(self, vid):
        """Crop the given video at specified timestamp and output size.
        the video is expected to be torch Tensor and have [..., T, H, W] shape,
        where ... means an arbitrary number of leading dimensions.

        Args:
            vid (Tensor): Video to be cropped.

        Returns:
            Tensor: Cropped video.
        """

        i, t = self.get_params(vid, self.size)

        return vid[..., i:i + t, :, :]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"
    

@register_transform("CropInTime")
class CropInTime(ClassyTransform):
    """
    Crop a spatiotemporal tensor in time.
    The input is a tensor of shape (C, T, H, W) and output is a tensor of shape (C, T', H, W)
    where T' < T.
    """

    def __init__(self, output_size: int):
        """
        Returns a cropped version of the input tensor by randomly selecting a subset of length output_size.
        """
        self.output_size = output_size

        # add and initialize the time crop transform
        transform = RandomTemporalCrop(self.output_size)
        self.transform = transform

    def __call__(self, vid: Tensor) -> Tensor:
        """
        Args:
            vid (Tensor): Input tensor of shape (C, T, H, W)

        Returns:
            Tensor: Cropped tensor of shape (C, T', H, W)
        """
        return self.transform(vid)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CropInTime":
        """
        Instantiates CropInTime from configuration.

        Args:
            config (Dict): arguments for the transform

        Returns:
            CropInTime instance.
        """
        #### not really sure what this part does
        return cls(**config)
