from __future__ import annotations

import cv2
import kornia
import numpy as np


kornia_op = kornia.filters.canny


def opencv_op(
        input: np.ndarray,
        low_threshold: float,
        high_threshold: float,
        kernel_size: int,
) -> None:
    # simulate batch as sequential op
    if len(input.shape) == 3:
        input = input[None]

    # TODO: do this casting on the runner via config
    input = (input * 255).astype(np.uint8)
    low_threshold *= 255
    high_threshold *= 255

    for i in range(input.shape[0]):
        x = input[i]
        cv2.Canny(x, low_threshold, high_threshold, kernel_size)
