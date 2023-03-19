from itertools import product

import torch
import torch._dynamo as dynamo
import torch.utils.benchmark as benchmark
from kornia.geometry.quaternion import Quaternion


torch.set_float32_matmul_precision('high')
# torch._dynamo.config.verbose=True
# torch._dynamo.config.log_level = logging.DEBUG


def quat_mm_kornia_eager_old(q1: Quaternion, q2: Quaternion):
    return q1 * q2


def quat_mm_kornia_eager_new(q1: Quaternion, q2: Quaternion):
    w0, x0, y0, z0 = q1.w, q1.x, q1.y, q1.z
    w1, x1, y1, z1 = q2.w, q2.x, q2.y, q2.z
    out = torch.stack(
        (
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        ), -1,
    )
    return out


def quat_mm_kornia_eager_mat(q1: Quaternion, q2: Quaternion):
    return q1.matrix() @ q2.matrix()


def quat_mm_kornia_eager_numpy(q1: Quaternion, q2: Quaternion):
    return q1.matrix().numpy() @ q2.matrix().numpy()


@dynamo.optimize('inductor')
def quat_mm_kornia_dynamo_old(q1: Quaternion, q2: Quaternion):
    return quat_mm_kornia_eager_old(q1, q2)


@dynamo.optimize('inductor')
def quat_mm_kornia_dynamo_new(q1: Quaternion, q2: Quaternion):
    return quat_mm_kornia_eager_new(q1, q2)


@dynamo.optimize('inductor')
def quat_mm_kornia_dynamo_mat(q1: Quaternion, q2: Quaternion):
    return quat_mm_kornia_eager_mat(q1, q2)


# Compare takes a list of measurements which we'll save in results.
results = []

batch_sizes = [1, 2, 5]
threads = [1, 4, 8]

backends = ['eager', 'dynamo']
versions = ['old', 'new', 'mat', 'numpy']
devices = ['cpu']
# devices = ["cpu", "cuda"]
dtypes = ['float32', 'float64']


def convert_to(data, device: str, dtype: str):
    return data.to(torch.device(device), eval(f'torch.{dtype}'))


def generate_sample(batch_size, device: str, dtype: str):
    quat1 = Quaternion.identity(batch_size).requires_grad_(False)
    quat2 = Quaternion.identity(batch_size).requires_grad_(False)
    # to device/dtype
    quat1 = convert_to(quat1, device, dtype)
    quat2 = convert_to(quat2, device, dtype)
    return quat1, quat2


for b, num_threads in product(batch_sizes, threads):
    # label and sub_label are the rows
    # description is the column
    label = 'Quaternion  multiplication'
    sub_label = f'[{b}]'
    for backend, device, dtype, version in product(backends, devices, dtypes, versions):
        if version == 'numpy' and backend == 'dynamo':
            continue
        base_desc = f'{backend}_{version}'  # e.g. eager_old
        base_fcn = f'quat_mm_kornia_{base_desc}'
        q1, q2 = generate_sample(b, device, dtype)  # move data to device
        stmt = f'{base_fcn}(q1,q2)'
        globals = {'q1': q1, 'q2': q2}
        results.append(
            benchmark.Timer(
                stmt=stmt,
                setup=f'from __main__ import {base_fcn}',
                globals=globals,
                num_threads=num_threads,
                label=label,
                sub_label=f'{sub_label}[{dtype}][{device}]',
                description=f'{base_desc}',
            ).blocked_autorange(min_run_time=1),
        )

compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.colorize()
compare.print()
