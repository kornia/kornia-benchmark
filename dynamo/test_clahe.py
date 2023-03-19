from itertools import product

import cv2
import kornia
import torch
import torch._dynamo as dynamo
import torch.utils.benchmark as benchmark

torch.set_float32_matmul_precision('high')
# torch._dynamo.config.verbose=True
# torch._dynamo.config.log_level = logging.DEBUG

torch_dynamo_optimize = dynamo.optimize('inductor')

op = kornia.enhance.equalize_clahe
op_dyn = torch_dynamo_optimize(op)

# simulate batch as sequential op
# NOTE: we include the data transfer because eventually as op included in the pipeline


def opencv(x):
    # NOTE: we should pass the images in numpy and in the proper range for a fair comparison.
    y_np = []
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
    if len(x.shape) == 3:
        x = x[None]
    for i in range(x.shape[0]):
        # equalize channels independently as gray scale images
        equ00 = torch.tensor(clahe.apply(kornia.utils.tensor_to_image(x[0].mul(255).clamp(0, 255).byte())))
        equ01 = torch.tensor(clahe.apply(kornia.utils.tensor_to_image(x[1].mul(255).clamp(0, 255).byte())))
        equ02 = torch.tensor(clahe.apply(kornia.utils.tensor_to_image(x[2].mul(255).clamp(0, 255).byte())))
        y_np.append(torch.stack([equ00, equ01, equ02]))


# Compare takes a list of measurements which we'll save in results.
results = []

batch_sizes = [None, 2]
resolution = [32, 64]
threads = [1, 4]

for b, n in product(batch_sizes, resolution):
    # label and sub_label are the rows
    # description is the column
    label = 'Clahe'
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
            ).blocked_autorange(min_run_time=1),
        )
        if torch.cuda.is_available():
            results.append(
                benchmark.Timer(
                    stmt='op(image)',
                    setup='from __main__ import op',
                    globals={'image': x.cuda()},
                    num_threads=num_threads,
                    label=label,
                    sub_label=sub_label,
                    description='eager_cuda',
                ).blocked_autorange(min_run_time=1),
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
            ).blocked_autorange(min_run_time=1),
        )
        if torch.cuda.is_available():
            results.append(
                benchmark.Timer(
                    stmt='op_dyn(image)',
                    setup='from __main__ import op_dyn',
                    globals={'image': x.cuda()},
                    num_threads=num_threads,
                    label=label,
                    sub_label=sub_label,
                    description='dynamo_cuda',
                ).blocked_autorange(min_run_time=1),
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
            ).blocked_autorange(min_run_time=1),
        )

compare = benchmark.Compare(results)
compare.print()
