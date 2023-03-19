from __future__ import annotations

import cv2
import kornia
import numpy as np
from kornia.core import Tensor


def kornia_op(
        input: Tensor,
        kernel_size: int,
        sigma: tuple[float, float],
) -> None:

    kornia.filters.unsharp_mask(input, kernel_size, tuple(sigma))


def opencv_op(
        input: np.ndarray,
        kernel_size: int,
        sigma: tuple[float, float],
) -> None:
    # simulate batch as sequential op
    if len(input.shape) == 3:
        input = input[None]

    k = (kernel_size, kernel_size)
    for i in range(input.shape[0]):
        x = input[i]
        data_blur = cv2.GaussianBlur(x, k, sigma[0], sigma[1])
        _ = x + (x - data_blur)
