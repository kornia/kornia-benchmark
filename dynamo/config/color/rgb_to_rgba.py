import cv2
import kornia
import numpy as np

kornia_op = kornia.color.rgb_to_rgba


def opencv_op(
        input: np.ndarray,
        alpha_val: float,
) -> None:
    # simulate batch as sequential op
    if len(input.shape) == 3:
        input = input[None]

    if input.shape[1] == 3:
        input = np.swapaxes(input, 1, -1)

    for i in range(input.shape[0]):
        x = input[i]
        cv2.cvtColor(x, code=cv2.COLOR_RGB2RGBA)
