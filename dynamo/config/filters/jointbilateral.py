from __future__ import annotations

import cv2
import kornia
import numpy as np
from kornia.core import Tensor


def kornia_op(
        input: Tensor,
        guidance: Tensor,
        kernel_size: int,
        sigma_color: float,
        sigma_space: float,
        color_distance_type: str,
) -> None:
    sig_sp = (sigma_space, sigma_space)
    kornia.filters.bilateral.joint_bilateral_blur(
        input, guidance, kernel_size, sigma_color, sig_sp,
        color_distance_type=color_distance_type,
    )


def opencv_op(
        input: np.ndarray,
        guidance: np.ndarray,
        kernel_size: int,
        sigma_color: float,
        sigma_space: float,
        color_distance_type: str,
) -> None:
    # simulate batch as sequential op
    if len(input.shape) == 3:
        input = input[None]

    # TODO: figure out to do this on the runner
    guidance = np.moveaxis(guidance, 1, -1)

    for i in range(input.shape[0]):
        x = input[i]
        g = guidance[i]
        cv2.ximgproc.jointBilateralFilter(
            g, x, kernel_size, sigma_color, sigma_space,
        )
