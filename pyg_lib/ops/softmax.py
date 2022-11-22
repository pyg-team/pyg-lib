from typing import Optional

import torch
from torch import Tensor

from pyg_lib._triton import tl, triton


@triton.jit
def softmax_kernel(inputs_ptr, ptr_ptr, out_ptr, M, N, num_segments, **meta):
    block_start_segment = tl.program_id(axis=0) * meta['BLOCK_SIZE_SEGMENT']
    block_start_N = tl.program_id(axis=1) * meta['BLOCK_SIZE_N']

    ptr_offset = block_start_segment + tl.arange(0, meta['BLOCK_SIZE_SEGMENT'])
    ptr_mask = ptr_offset < num_segments

    N_offset = block_start_N + tl.arange(0, meta['BLOCK_SIZE_N'])
    N_mask = N_offset < N

    start = tl.load(ptr_ptr + ptr_offset, mask=ptr_mask, other=M)
    end = tl.load(ptr_ptr + 1 + ptr_offset, mask=ptr_mask, other=M)
    count = end - start

    # tl.
    a = 10
    max_value = tl.max(count, axis=0)
    M_offset = tl.arange(0, a)
    # M_mask = M_offset < count

    inputs_offset = (N * start[:, None, None] + N * M_offset[None, :, None] +
                     N_offset[None, None, :])
    inputs_mask = (ptr_mask[:, None, None] &
                   (M_offset[None, :, None] < count[:, None, None])
                   & N_mask[:, None, None])

    inputs = tl.load(inputs_ptr + inputs_offset, mask=inputs_mask,
                     other=float('-inf'))

    # x = inputs - 2  # tl.max(inputs, axis=1)[:, None, :]
    x = inputs
    # x = x - tl.max(inputs, axis=1)[:, None, :]
    x = tl.exp(x)
    x = x / tl.sum(x, axis=1)[:, None, :]

    tl.store(out_ptr + inputs_offset, x, mask=inputs_mask)

    # ptr_offset = ptr_block_start + tl.arange(0, meta['SEGMENT_BLOCK_SIZE'])
    # ptr_mask = ptr_offset < num_segments

    # ptr1 = tl.load(ptr_ptr + ptr_offset, mask=ptr_mask, other=1000000)
    # ptr2 = tl.load(ptr_ptr + ptr_offset + 1, mask=ptr_mask, other=1000000)
    # count = ptr2 - ptr1
    # # max_count = tl.max(ptr2 - ptr1, axis=0)
    # # max_count = tl.multiple_of(max_count, 8)
    # max_count = 10  # TODO
    # M_offset = tl.arange(0, max_count)

    # N_block_start = tl.program_id(axis=1) * meta['BLOCK_SIZE_N']
    # N_offset = N_block_start + tl.arange(0, meta['BLOCK_SIZE_N'])

    # x_offset = (N * ptr1[:, None, None] + N * M_offset[None, :, None] +
    #             N_offset[None, None, :])
    # x_mask = ((ptr1[:, None, None] < M) &
    #           (M_offset[None, :, None] < count[:, None, None]) &
    #           (N_offset[None, None, :] < N))

    # x = tl.load(x_ptr + x_offset, mask=x_mask, other=float('-inf'))
    # x = x - tl.max(x, axis=1)[:, None, :]
    # x = tl.exp(x)
    # out = x / tl.sum(x, axis=1)[:, None, :]

    # tl.store(out_ptr + x_offset, out, mask=x_mask)


def softmax(
    inputs: Tensor,
    ptr: Tensor,
    out: Optional[Tensor] = None,
) -> Tensor:
    if out is None:
        out = torch.empty_like(inputs)
    out.resize_(inputs.size())

    out.fill_(-1)  # TODO

    assert inputs.dim() == 2 and inputs.is_cuda and inputs.is_contiguous()
    assert ptr.dim() == 1 and ptr.is_cuda and ptr.is_contiguous()
    assert out.dim() == 2 and out.is_cuda and out.is_contiguous()

    (M, N), num_segments = inputs.size(), ptr.numel() - 1

    grid = lambda meta: (
        triton.cdiv(num_segments, meta['BLOCK_SIZE_SEGMENT']),
        triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    softmax_kernel[grid](inputs, ptr, out, M, N, num_segments,
                         BLOCK_SIZE_SEGMENT=1, BLOCK_SIZE_N=1)

    return out
