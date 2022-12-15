from itertools import product

import torch
import torch._dynamo as dynamo
import torch.utils.benchmark as benchmark

import kornia as K
from kornia.geometry.liegroup import Se3


torch.set_float32_matmul_precision('high')
# torch._dynamo.config.verbose=True
# torch._dynamo.config.log_level = logging.DEBUG

@torch.inference_mode()
def op_matmul_eager_at(m1, m2):
    m3 = m1 @ m2
    return m3

@dynamo.optimize("inductor")
def op_matmul_dynamo_at(m1, m2):
    m3 = m1 @ m2
    return m3

def op_matmul_numpy_at(m1, m2):
    m3 = m1 @ m2
    return m3


# Compare takes a list of measurements which we'll save in results.
results = []

batch_sizes = [1, 2]
dims = [3, 4]
threads = [1, 4]

optimizers = ["eager", "dynamo", "numpy"]
methods = ["at"]
#devices = ["cpu"]
devices = ["cpu", "cuda"]
dtypes = ["float32"]
#dtypes = ["float32", "float64"]


def convert_to(data, device: str, dtype: str):
    return data.to(torch.device(device), eval(f"torch.{dtype}"))


def generate_sample(batch_size, dim, device: str, dtype: str, backend):
    mat1 = torch.rand(batch_size, dim, dim)
    mat2 = torch.rand(batch_size, dim, dim)
    # to device/dtype
    mat1 = convert_to(mat1, device, dtype)
    mat2 = convert_to(mat2, device, dtype)
    if backend == "numpy":
        mat1 = mat1.cpu().numpy()
        mat2 = mat2.cpu().numpy()
    return mat1, mat2


for b, dim, num_threads in product(batch_sizes, dims, threads):
    # label and sub_label are the rows
    # description is the column
    label = 'Matrix multiplication'
    sub_label = f'[{b}][{dim}x{dim}]'
    for optimizer, device, dtype, method in  product(optimizers, devices, dtypes, methods):
        if optimizer == "numpy" and device == "cuda":
            continue
        base_desc = f'{optimizer}_{method}'
        base_fcn = f'op_matmul_{base_desc}'
        m1, m2 = generate_sample(b, dim, device, dtype, optimizer)  # move data to device
        stmt = f'{base_fcn}(m1,m2)'
        globals = {'m1': m1, 'm2': m2}
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

compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.colorize()
compare.print()