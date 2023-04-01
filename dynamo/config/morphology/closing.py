import cv2
import kornia
import numpy as np

kornia_op = kornia.morphology.closing


def opencv_op(
        input: np.ndarray,
        kernel: np.ndarray,
        *args,
        **kwargs
) -> None:
    # simulate batch as sequential op
    kernel = kernel.astype(np.uint8)
    if len(input.shape) == 3:
        input = input[None]

    for i in range(input.shape[0]):
        x = input[i]
        cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel)
