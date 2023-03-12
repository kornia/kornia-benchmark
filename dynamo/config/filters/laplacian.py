from __future__ import annotations

import cv2
import kornia
import numpy as np


kornia_op = kornia.filters.laplacian


def opencv_op(
        input: np.ndarray,
        kernel_size: int,
) -> None:
    # simulate batch as sequential op
    if len(input.shape) == 3:
        input = input[None]

    # TODO: do this casting on the runner via config
    input = (input * 255).astype(np.uint8)

    for i in range(input.shape[0]):
        x = input[i]
        cv2.Laplacian(x, cv2.CV_16S, ksize=kernel_size)
