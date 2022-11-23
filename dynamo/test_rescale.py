from itertools import product
import logging

import cv2
import torch
import torch._dynamo as dynamo
import torch.utils.benchmark as benchmark

import kornia


torch.set_float32_matmul_precision('high')
# torch._dynamo.config.verbose=True
# torch._dynamo.config.log_level = logging.DEBUG

torch_dynamo_optimize = dynamo.optimize("inductor")

op = kornia.geometry.transform.rescale
op_dyn = torch_dynamo_optimize(op)

# simulate batch as sequential op

def opencv(input, factor):
    if len(input.shape) == 3:
        input = input[None]
    for i in range(input.shape[0]):
        x_np = kornia.tensor_to_image(input[i])
        h, w = x.shape[:2]
        h_new = int(h * factor[0])
        w_new = int(w * factor[1])
        y_np = cv2.pyrDown(x_np, (w_new, h_new))


# Compare takes a list of measurements which we'll save in results.
results = []

batch_sizes = [1, 2, 5, 9]
resolution = [32, 64, 128, 256, 512]
factor = [0.5, 1.5]
threads = [1, 4]

for b, n, f in product(batch_sizes, resolution, factor):
    # label and sub_label are the rows
    # description is the column
    label = 'Rescale'
    sub_label = f'[{b}, {n}, {f}]'
    x = torch.ones((3, n, n))
    if b is not None:
        x = x[None].repeat(b, 1, 1, 1)
    for num_threads in threads:
        results.append(
            benchmark.Timer(
                stmt='op(input, factor)',
                setup='from __main__ import op',
                globals={'input': x, 'factor': (f, f)},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='eager_cpu',
            ).blocked_autorange(min_run_time=1)
        )
        results.append(
            benchmark.Timer(
                stmt='op(input, factor)',
                setup='from __main__ import op',
                globals={'input': x.cuda(), 'factor': (f, f)},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='eager_cuda',
            ).blocked_autorange(min_run_time=1)
        )
        results.append(
            benchmark.Timer(
                stmt='op_dyn(input, factor)',
                setup='from __main__ import op_dyn',
                globals={'input': x, 'factor': (f, f)},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='dynamo_cpu',
            ).blocked_autorange(min_run_time=1)
        )
        results.append(
            benchmark.Timer(
                stmt='op_dyn(input, factor)',
                setup='from __main__ import op_dyn',
                globals={'input': x.cuda(), 'factor': (f, f)},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='dynamo_cuda',
            ).blocked_autorange(min_run_time=1)
        )
        results.append(
            benchmark.Timer(
                stmt='opencv(input, factor)',
                setup='from __main__ import opencv',
                globals={'input': x.byte(), 'factor': (f, f)},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='opencv',
            ).blocked_autorange(min_run_time=1)
        )

compare = benchmark.Compare(results)
compare.print()