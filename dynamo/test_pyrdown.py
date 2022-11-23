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

op = kornia.geometry.transform.pyrdown
op_dyn = torch_dynamo_optimize(op)

# simulate batch as sequential op

def opencv(x):
    if len(x.shape) == 3:
        x = x[None]
    for i in range(x.shape[0]):
        x_np = kornia.tensor_to_image(x[i])
        y_np = cv2.pyrDown(x_np)


# Compare takes a list of measurements which we'll save in results.
results = []

batch_sizes = [1, 2, 5, 9]
resolution = [32, 64, 128, 256, 512]
threads = [1, 4]

for b, n in product(batch_sizes, resolution):
    # label and sub_label are the rows
    # description is the column
    label = 'PyrDown'
    sub_label = f'[{b}, {n}]'
    x = torch.ones((3, n, n))
    if b is not None:
        x = x[None].repeat(b, 1, 1, 1)
    for num_threads in threads:
        results.append(
            benchmark.Timer(
                stmt='op(image)',
                setup='from __main__ import op',
                globals={'image': x},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='eager_cpu',
            ).blocked_autorange(min_run_time=1)
        )
        results.append(
            benchmark.Timer(
                stmt='op(image)',
                setup='from __main__ import op',
                globals={'image': x.cuda()},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='eager_cuda',
            ).blocked_autorange(min_run_time=1)
        )
        results.append(
            benchmark.Timer(
                stmt='op_dyn(image)',
                setup='from __main__ import op_dyn',
                globals={'image': x},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='dynamo_cpu',
            ).blocked_autorange(min_run_time=1)
        )
        results.append(
            benchmark.Timer(
                stmt='op_dyn(image)',
                setup='from __main__ import op_dyn',
                globals={'image': x.cuda()},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='dynamo_cuda',
            ).blocked_autorange(min_run_time=1)
        )
        results.append(
            benchmark.Timer(
                stmt='opencv(image)',
                setup='from __main__ import opencv',
                globals={'image': x.byte()},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='opencv',
            ).blocked_autorange(min_run_time=1)
        )

compare = benchmark.Compare(results)
compare.print()