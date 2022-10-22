from typing import Optional

import torch
import triton
# import triton.language as tl
from torch import Tensor


@triton.jit
def softmax_kernel(x_ptr, ptr, out_ptr, numel, **meta):
    # pid = tl.program_id(axis=0)
    # block_start = pid * meta['BLOCK_SIZE']

    # offsets = block_start + tl.arange(0, meta['BLOCK_SIZE'])
    # mask = offsets < numel

    # x = tl.load(x_ptr + offsets, mask=mask)
    # y = tl.load(y_ptr + offsets, mask=mask)

    # output = x + y

    # tl.store(out_ptr + offsets, output, mask=mask)
    pass


def softmax(
    inputs: Tensor,
    ptr: Tensor,
    out: Optional[Tensor] = None,
) -> Tensor:
    if out is None:
        out = torch.empty_like(inputs)
    assert inputs.is_cuda and ptr.is_cuda and out.is_cuda

    grid = lambda meta: (triton.cdiv(inputs.numel(), meta['BLOCK_SIZE']), )
    softmax_kernel[grid](inputs, ptr, out, inputs.numel(), BLOCK_SIZE=1024)
    return out
