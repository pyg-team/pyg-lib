from typing import Optional

import torch
from torch import Tensor

from pyg_lib._triton import tl, triton


@triton.jit
def softmax_kernel(x_ptr, ptr_ptr, out_ptr, M, N, num_segments, **meta):
    ptr_block_start = tl.program_id(axis=0) * meta['SEGMENT_BLOCK_SIZE']
    ptr_offset = ptr_block_start + tl.arange(0, meta['SEGMENT_BLOCK_SIZE'])
    ptr_mask = ptr_offset < num_segments

    ptr1 = tl.load(ptr_ptr + ptr_offset, mask=ptr_mask, other=1000000)
    ptr2 = tl.load(ptr_ptr + ptr_offset + 1, mask=ptr_mask, other=1000000)
    count = ptr2 - ptr1
    # max_count = tl.max(ptr2 - ptr1, axis=0)
    # max_count = tl.multiple_of(max_count, 8)
    max_count = 10  # TODO
    M_offset = tl.arange(0, max_count)

    N_block_start = tl.program_id(axis=1) * meta['BLOCK_SIZE_N']
    N_offset = N_block_start + tl.arange(0, meta['BLOCK_SIZE_N'])

    # M_mask = M_offset[None, :] < count[:, None]

    x_offset = (N * ptr1[:, None, None] + N * M_offset[None, :, None] +
                N_offset[None, None, :])
    x_mask = ((ptr1[:, None, None] < M) &
              (M_offset[None, :, None] < count[:, None, None]) &
              (N_offset[None, None, :] < N))

    x = tl.load(x_ptr + x_offset, mask=x_mask, other=float('-inf'))
    x = x - tl.max(x, axis=1)[:, None, :]
    x = tl.exp(x)
    out = x / tl.sum(x, axis=1)[:, None, :]

    tl.store(out_ptr + x_offset, out, mask=x_mask)


def softmax(
    inputs: Tensor,
    ptr: Tensor,
    out: Optional[Tensor] = None,
) -> Tensor:
    if out is None:
        out = torch.empty_like(inputs)
    out.resize_(inputs.size())

    out.fill_(-1)

    assert inputs.dim() == 2 and inputs.is_cuda and inputs.is_contiguous()
    assert ptr.dim() == 1 and ptr.is_cuda and ptr.is_contiguous()
    assert out.dim() == 2 and out.is_cuda and out.is_contiguous()

    (M, N), num_segments = inputs.size(), ptr.numel() - 1
    print('M', M, 'N', N, 'num_segments', num_segments)

    grid = lambda meta: (
        triton.cdiv(num_segments, meta['SEGMENT_BLOCK_SIZE']),
        triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    softmax_kernel[grid](inputs, ptr, out, M, N, num_segments,
                         SEGMENT_BLOCK_SIZE=1, BLOCK_SIZE_N=1)
    return out
