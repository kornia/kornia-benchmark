from __future__ import annotations

import cv2
import kornia
import numpy as np


kornia_op = kornia.filters.motion_blur


def opencv_op(
        input: np.ndarray,
        kernel_size: int,
        angle: float,
        direction: float = 0.0,
) -> None:
    # simulate batch as sequential op
    if len(input.shape) == 3:
        input = input[None]

    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    k[(kernel_size-1) // 2, :] = np.ones(kernel_size, dtype=np.float32)
    k = cv2.warpAffine(
        k, cv2.getRotationMatrix2D(
            (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5), angle, 1.0,
        ),
        (kernel_size, kernel_size),
    )
    k = k * (1.0 / np.sum(k))

    for i in range(input.shape[0]):
        x = input[i]
        cv2.filter2D(x, -1, k)
