from itertools import product

import cv2
import kornia
import numpy as np
import requests
import torch
import torch._dynamo as dynamo
import torch.utils.benchmark as benchmark


torch.set_float32_matmul_precision('high')
optimizer = dynamo.optimize('inductor')


kornia_op = kornia.morphology.closing
kornia_op_optimized = optimizer(kornia_op)


def opencv_op(
        input: np.ndarray,
        kernel: np.ndarray,
        *args,
        **kwargs
) -> None:
    # simulate batch as sequential op
    for i in range(input.shape[0]):
        x = input[i]
        cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel)


url_img = 'https://raw.githubusercontent.com/kornia/data/main/baby_giraffe.png'

response = requests.get(url_img).content
nparr = np.frombuffer(response, np.uint8)
img_default = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)[..., :3]
img_default_t = kornia.utils.image_to_tensor(img_default, keepdim=False)
img_default_t = img_default_t.float() / 255.0

res = img_default.shape[:2]

k_sizes = [3]
num_threads = 1
b_sizes = [1, 3]
engines = ['unfold', 'convolution']
dev_cuda = torch.device('cuda')
results = []

for bs, kernel_size, engine in product(b_sizes, k_sizes, engines):
    sub_label = f'[{bs}, {res}, {kernel_size}, {engine}]'
    print(sub_label)

    # batch image
    img = np.repeat(img_default[None, ...], bs, 0)
    img_t = img_default_t.repeat(bs, 1, 1, 1)

    # generate kernels
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    kernel_t = torch.from_numpy(kernel).to(dtype=img_t.dtype)

    # try
    print('\ttesting opencv')
    opencv_op(img, kernel)

    print('\ttesting eager')
    kornia_op(img_t, kernel_t)

    print('\ttesting dynamo')
    kornia_op_optimized(img_t, kernel_t)

    # bench opencv uint8
    print('\tbench opencv uint8...')
    results.append(
        benchmark.Timer(
            stmt='opencv_op(image, kernel)',
            setup='from __main__ import opencv_op',
            globals={'image': img, 'kernel': kernel},
            num_threads=num_threads,
            label='closing',
            sub_label=sub_label,
            description='opencv_uint8',
        ).blocked_autorange(min_run_time=1),
    )

    # bench opencv float32
    print('\tbench opencv float32...')
    results.append(
        benchmark.Timer(
            stmt='opencv_op(image, kernel)',
            setup='from __main__ import opencv_op',
            globals={
                'image': (img / 255.0).astype(np.float32),
                'kernel': kernel,
            },
            num_threads=num_threads,
            label='closing',
            sub_label=sub_label,
            description='opencv_float32',
        ).blocked_autorange(min_run_time=1),
    )

    # bench eager cpu
    print('\tbench eager cpu...')
    results.append(
        benchmark.Timer(
            stmt='kornia_op(image, kernel, engine=engine)',
            setup='from __main__ import kornia_op',
            globals={
                'image': img_t,
                'kernel': kernel_t,
                'engine': engine,
            },
            num_threads=num_threads,
            label='closing',
            sub_label=sub_label,
            description='eager_cpu',
        ).blocked_autorange(min_run_time=1),
    )

    # bench eager cuda
    print('\tbench eager cuda...')
    results.append(
        benchmark.Timer(
            stmt='kornia_op(image, kernel, engine=engine)',
            setup='from __main__ import kornia_op',
            globals={
                'image': img_t.to(device=dev_cuda),
                'kernel': kernel_t.to(device=dev_cuda),
                'engine': engine,
            },
            num_threads=num_threads,
            label='closing',
            sub_label=sub_label,
            description='eager_cuda',
        ).blocked_autorange(min_run_time=1),
    )

    # bench dynamo cpu
    print('\tbench dynamo cpu...')
    results.append(
        benchmark.Timer(
            stmt='kornia_op_optimized(image, kernel,engine=engine)',
            setup='from __main__ import kornia_op_optimized',
            globals={
                'image': img_t,
                'kernel': kernel_t,
                'engine': engine,
            },
            num_threads=num_threads,
            label='closing',
            sub_label=sub_label,
            description='dynamo_cpu',
        ).blocked_autorange(min_run_time=1),
    )

    # bench dynamo cuda
    print('\tbench dynamo cuda...')
    results.append(
        benchmark.Timer(
            stmt='kornia_op_optimized(image, kernel,engine=engine)',
            setup='from __main__ import kornia_op_optimized',
            globals={
                'image': img_t.to(device=dev_cuda),
                'kernel': kernel_t.to(device=dev_cuda),
                'engine': engine,
            },
            num_threads=num_threads,
            label='closing',
            sub_label=sub_label,
            description='dynamo_cuda',
        ).blocked_autorange(min_run_time=1),
    )

compare = benchmark.Compare(results)
compare.print()
