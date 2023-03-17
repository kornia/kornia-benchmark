from __future__ import annotations

import cv2
import kornia
import numpy as np
from kornia.core import Tensor


def kornia_op(
        input: Tensor,
        kernel_size: int,
        sigma: list[float],
) -> None:
    kornia.filters.gaussian_blur2d(input, kernel_size, tuple(sigma))


def opencv_op(
        input: np.ndarray,
        kernel_size: int,
        sigma: list[float],
) -> None:
    # simulate batch as sequential op
    if len(input.shape) == 3:
        input = input[None]

    k = (kernel_size, kernel_size)
    for i in range(input.shape[0]):
        x = input[i]
        cv2.GaussianBlur(x, k, sigma[0], sigma[1])
