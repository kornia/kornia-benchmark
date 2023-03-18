# Example based on https://pytorch.org/docs/master/dynamo/troubleshooting.html#minifying-backend-compiler-errors
import kornia
import torch
import torch._dynamo as dynamo

torch.set_float32_matmul_precision('high')
torch_dynamo_optimize = dynamo.optimize('inductor')

# torch._dynamo.config.repro_level = 4
torch._dynamo.config.repro_after = 'dynamo'  # or "aot"

input_tensor = torch.rand(
    1, 4, 5, 6,
    device=torch.device('cuda'),
    dtype=torch.float64,
)
M = torch.eye(2, 3, device=input_tensor.device, dtype=input_tensor.dtype)[None]
dsize = (4, 2)
fill_value = torch.zeros(
    3,
    device=input_tensor.device,
    dtype=input_tensor.dtype,
)

kornia_op = kornia.geometry.transform.warp_affine
op = torch_dynamo_optimize(kornia_op)

args = [input_tensor, M, dsize, 'bilinear', 'zeros', True, fill_value]

with torch.cuda.amp.autocast(enabled=False):
    ref = kornia_op(*args)
    res = op(*args)
