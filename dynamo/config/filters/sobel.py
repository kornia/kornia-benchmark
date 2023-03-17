from __future__ import annotations

import cv2
import kornia
import numpy as np


kornia_op = kornia.filters.sobel


def opencv_op(
        input: np.ndarray,
) -> None:
    # simulate batch as sequential op
    if len(input.shape) == 3:
        input = input[None]

    for i in range(input.shape[0]):
        x = input[i]
        cv2.Sobel(x, cv2.CV_64F, dx=1, dy=1, ksize=3)
