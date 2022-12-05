import torch
from torch import Tensor

from pyg_lib._triton import tl, triton


@triton.jit
def broadcast_sub_kernel(inputs_ptr, other_ptr, index_ptr, out_ptr, num_feats,
                         numel, **meta):

    pid = tl.program_id(axis=0)
    block_start = pid * meta['BLOCK_SIZE']

    offsets = block_start + tl.arange(0, meta['BLOCK_SIZE'])
    mask = offsets < numel
    inputs = tl.load(inputs_ptr + offsets, mask=mask)

    index_offsets = offsets // num_feats
    index = tl.load(index_ptr + index_offsets, mask=mask)

    other_offsets = (num_feats * index) + (offsets % num_feats)
    other = tl.load(other_ptr + other_offsets, mask=mask)

    out = inputs - other
    tl.store(out_ptr + offsets, out, mask=mask)


def broadcast_sub(inputs: Tensor, other: Tensor, index: Tensor) -> Tensor:
    assert inputs.is_cuda and inputs.is_contiguous()
    assert other.is_cuda and other.is_contiguous()
    assert index.is_cuda and index.is_contiguous()

    assert inputs.dim() == 2 and other.dim() == 2
    assert inputs.size(-1) == other.size(-1)
    assert inputs.size(0) == index.size(0)

    out = torch.empty_like(inputs)

    grid = lambda meta: (triton.cdiv(inputs.numel(), meta['BLOCK_SIZE']), )
    broadcast_sub_kernel[grid](inputs, other, index, out, inputs.size(-1),
                               inputs.numel(), BLOCK_SIZE=1024)
    return out
