from typing import Tuple

import cv2
import kornia
import numpy as np

kornia_op = kornia.geometry.transform.rescale


def opencv_op(
        input: np.ndarray,
        factor: Tuple[float, float],
        interpolation: int = cv2.INTER_LINEAR,
) -> None:
    # simulate batch as sequential op
    if len(input.shape) == 3:
        input = input[None]

    for i in range(input.shape[0]):
        x = input[i]
        h, w = x.shape[:2]
        h_new = int(h * factor[0])
        w_new = int(w * factor[1])
        cv2.resize(x, (w_new, h_new), interpolation=interpolation)
