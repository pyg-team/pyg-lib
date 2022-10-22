import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, numel, **meta):
    pid = tl.program_id(axis=0)
    block_start = pid * meta['BLOCK_SIZE']

    offsets = block_start + tl.arange(0, meta['BLOCK_SIZE'])
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    tl.store(out_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and out.is_cuda
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, out, x.numel(), BLOCK_SIZE=1024)
    return out


def test_triton():
    x = torch.rand(100, device='cuda')
    y = torch.rand(100, device='cuda')
    assert torch.allclose(x + y, add(x, y))
