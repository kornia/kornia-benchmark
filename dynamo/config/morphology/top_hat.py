import cv2
import kornia
import numpy as np

kornia_op =kornia.morphology.top_hat


def opencv_op(
        input: np.ndarray,
        kernel: np.ndarray
) -> None:
    # simulate batch as sequential op
    input = input.astype(np.uint8)
    kernel = kernel.astype(np.uint8)
    if len(input.shape) == 3:
        input = input[None]
    if len(kernel.shape) == 3:
        kernel = kernel[None]

    for i in range(input.shape[0]):
        x = input[i]
        k = kernel[i]
        cv2.morphologyEx(x, cv2.MORPH_TOPHAT, k)
