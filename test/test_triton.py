import torch
from torch import Tensor

from pyg_lib._triton import tl, triton
from pyg_lib.testing import onlyCUDA, onlyTriton


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    tl.store(out_ptr + offsets, output, mask=mask)


def add(x: Tensor, y: Tensor) -> Tensor:
    out = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and out.is_cuda
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, out, x.numel(), BLOCK_SIZE=1024)
    return out


@onlyCUDA
@onlyTriton
def test_triton():
    x = torch.rand(100, device='cuda')
    y = torch.rand(100, device='cuda')
    assert torch.allclose(x + y, add(x, y))
