from itertools import product

import cv2
import numpy as np
import torch
import torch._dynamo as dynamo
import torch.utils.benchmark as benchmark

import kornia as K
import torch.nn.functional as F


torch.set_float32_matmul_precision('high')
# torch._dynamo.config.verbose=True
# torch._dynamo.config.log_level = logging.DEBUG


def remap_opencv(image, map_x, map_y, batch_size):
    # simulate batch size
    outs = []
    for _ in range(batch_size):
        out = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        outs.append(out)
    return np.stack(outs)


def remap_kornia_eager_new(image, map_xy):
    return F.grid_sample(image, map_xy, mode="bilinear")


def remap_kornia_eager_old(image, map_x, map_y):
    return K.geometry.transform.remap(
        image,
        map_x,
        map_y,
        mode="bilinear",
        normalized_coordinates=True)


@torch.jit.script
def remap_kornia_jit_old(image, map_x, map_y):
    return remap_kornia_eager_old(image, map_x, map_y)


@dynamo.optimize("inductor")
def remap_kornia_dynamo_old(image, map_x, map_y):
    return remap_kornia_eager_old(image, map_x, map_y)


@torch.jit.script
def remap_kornia_jit_new(image, map_xy):
    return remap_kornia_eager_new(image, map_xy)


@dynamo.optimize("inductor")
def remap_kornia_dynamo_new(image, map_xy):
    return remap_kornia_eager_new(image, map_xy)


# Compare takes a list of measurements which we'll save in results.
results = []

batch_sizes = [1, 2]
channels = [1, 3]
# resolutions = [(32, 64), (64, 128), (128, 224), (400, 640)]
resolutions = [(400, 640)]
threads = [1, 4, 8]

# backends = ["eager", "jit", "dynamo"]
backends = ["eager", "dynamo"]
versions = ["new", "old"]
devices = ["cpu", "cuda"]
dtypes = ["float16", "float32", "float64"]


def convert_to(data, device: str, dtype: str):
    return data.to(torch.device(device), eval(f"torch.{dtype}"))


def generate_sample(batch_size, channels, height, width, device: str, dtype: str, data_backend = "torch", normalized = True):
    img = torch.ones((batch_size, channels, height, width))
    base_grid = K.utils.create_meshgrid(height, width, normalized_coordinates=normalized)
    base_grid = base_grid.repeat(batch_size, 1, 1, 1)
    map_x = base_grid[..., 0]
    map_y = base_grid[..., 1]
    # to device/dtype
    img = convert_to(img, device, dtype)
    map_x = convert_to(map_x, device, dtype)
    map_y = convert_to(map_y, device, dtype)
    # to numpy (if needed)
    if data_backend == "numpy":
        img = K.utils.tensor_to_image(img[0])
        map_x = K.utils.tensor_to_image(map_x[0])
        map_y = K.utils.tensor_to_image(map_y[0])
    return  img, map_x, map_y


for b, ch, (h, w) in product(batch_sizes, channels, resolutions):
    # label and sub_label are the rows
    # description is the column
    label = 'Remap'
    sub_label = f'[{b}x{ch}x{h}x{w}]'
    for num_threads in threads:
        for bck, device, dtype, ver in  product(backends, devices, dtypes, versions):
            if device == "cpu" and dtype == "float16":
                continue  # RuntimeError: grid_sampler_2d_cpu not implemented for Half
            base_desc = f'{bck}_{ver}'  # e.g. eager_old
            base_fcn = f'remap_kornia_{base_desc}'
            image, map_x, map_y = generate_sample(b, ch, h, w, device, dtype)  # move data to device
            if ver == "old":
                stmt = f'{base_fcn}(image, map_x, map_y)'
                globals = {'image': image, 'map_x': map_x, 'map_y': map_y}
            elif ver == "new":
                stmt = f'{base_fcn}(image, map_xy)'
                globals = {'image': image, 'map_xy': torch.stack((map_x, map_y), -1)}
            results.append(
                benchmark.Timer(
                    stmt=stmt,
                    setup=f'from __main__ import {base_fcn}',
                    globals=globals,
                    num_threads=num_threads,
                    label=label,
                    sub_label=f'{sub_label}[{dtype}][{device}]',
                    description=f'{base_desc}',
                ).blocked_autorange(min_run_time=1)
            )
        # test case for opencv
        image, map_x, map_y = generate_sample(
            b, ch, h, w, device="cpu", dtype="float32", data_backend="numpy")  # move data to device
        results.append(
            benchmark.Timer(
                stmt='remap_opencv(image, map_x, map_y, batch_size)',
                setup='from __main__ import remap_opencv',
                globals={'image': image, 'map_x': map_x, 'map_y': map_y, 'batch_size': b},
                num_threads=num_threads,
                label=label,
                sub_label=f'{sub_label}[float32][cpu]',
                description='opencv_cpu',
            ).blocked_autorange(min_run_time=1)
        )

compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.colorize()
compare.print()