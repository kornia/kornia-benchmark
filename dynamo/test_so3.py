from itertools import product

import numpy as np
import torch
import torch._dynamo as dynamo
import torch.utils.benchmark as benchmark

import kornia as K
from kornia.geometry.liegroup.so3 import So3
import torch.nn.functional as F


torch.set_float32_matmul_precision('high')

def so3_mul_kornia_eager(r1: So3, r2: So3):
	return r1 * r2 

@dynamo.optimize("inductor")
def so3_mul_kornia_dynamo(r1: So3, r2: So3):
	return r1 * r2

def so3_mm_kornia_eager(r1: So3, r2: So3):
	return r1.matrix() @ r2.matrix() 

@dynamo.optimize("inductor")
def so3_mm_kornia_dynamo(r1: So3, r2: So3):
	return r1.matrix() @ r2.matrix() 

def so3_mm_kornia_numpy(r1: So3, r2: So3):
	return r1.matrix().numpy() @ r2.matrix().numpy() 

results = []

batch_sizes = [1, 2, 5]
threads = [1, 4, 8]

backends = ["eager", "dynamo", "numpy"]
methods = ["mul", "mm"]
devices = ["cpu"]
# devices = ["cpu", "cuda"]
dtypes = ["float32", "float64"]


def convert_to(data, device: str, dtype: str):
    return data.to(torch.device(device), eval(f"torch.{dtype}"))


def generate_sample(batch_size, device: str, dtype: str):
    r1 = So3.identity(batch_size).requires_grad_(False)
    r2 = So3.identity(batch_size).requires_grad_(False)
    r1 = convert_to(r1, device, dtype)
    r2 = convert_to(r2, device, dtype)
    return r1, r2


for b, num_threads in product(batch_sizes, threads):
    # label and sub_label are the rows
    # description is the column
    label = 'So3  multiplication'
    sub_label = f'[{b}]'
    for backend, method, device, dtype in  product(backends, methods, devices, dtypes):
        if backend == "numpy" and method == "mul":
            continue
        base_fcn = f'so3_{method}_kornia_{backend}'
        r1, r2 = generate_sample(b, device, dtype)  # move data to device
        stmt = f'{base_fcn}(r1,r2)'
        globals = {'r1': r1, 'r2': r2}
        results.append(
            benchmark.Timer(
                stmt=stmt,
                setup=f'from __main__ import {base_fcn}',
                globals=globals,
                num_threads=num_threads,
                label=label,
                sub_label=f'{sub_label}[{dtype}][{device}]',
                description=f'{base_fcn}',
            ).blocked_autorange(min_run_time=1)
        )

compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.colorize()
compare.print()