import cv2
import kornia
import numpy as np
from kornia.core import Tensor

kornia_op = kornia.color.bgr_to_rgb

KORNIA_OPS = {
    'bgr_to_rgb': kornia.color.bgr_to_rgb,
    'rgb_to_grayscale': kornia.color.rgb_to_grayscale,
    'rgb_to_hls': kornia.color.rgb_to_hls,
    'rgb_to_hsv': kornia.color.rgb_to_hsv,
    'rgb_to_lab': kornia.color.rgb_to_lab,
    'rgb_to_luv': kornia.color.rgb_to_luv,
    'rgb_to_xyz': kornia.color.rgb_to_xyz,
    'rgb_to_ycbcr': kornia.color.rgb_to_ycbcr,  # at opencv is ycrcb
    'rgb_to_yuv': kornia.color.rgb_to_yuv,
}

CV2_CODES = {
    'bgr_to_rgb': cv2.COLOR_BGR2RGB,
    'rgb_to_grayscale': cv2.COLOR_RGB2GRAY,
    'rgb_to_hls': cv2.COLOR_RGB2HLS,
    'rgb_to_hsv': cv2.COLOR_RGB2HSV,
    'rgb_to_lab': cv2.COLOR_RGB2LAB,
    'rgb_to_luv': cv2.COLOR_RGB2LUV,
    'rgb_to_xyz': cv2.COLOR_RGB2XYZ,
    'rgb_to_ycbcr': cv2.COLOR_RGB2YCrCb,
    'rgb_to_yuv': cv2.COLOR_RGB2YUV,
}


def kornia_op(input: Tensor, *args, **kwargs):
    mode = kwargs['mode']
    del kwargs['mode']

    KORNIA_OPS[mode](input, *args, **kwargs)


def opencv_op(input: np.ndarray, *args, **kwargs) -> None:
    mode = kwargs['mode']

    # simulate batch as sequential op
    if len(input.shape) == 3:
        input = input[None]

    if input.shape[1] == 3:
        input = np.swapaxes(input, 1, -1)

    for i in range(input.shape[0]):
        x = input[i]
        cv2.cvtColor(x, code=CV2_CODES[mode])
