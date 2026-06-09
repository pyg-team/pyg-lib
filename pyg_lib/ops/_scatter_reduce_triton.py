from typing import Tuple

from torch import Tensor

from pyg_lib._triton import load_triton

triton, tl = load_triton()


@triton.jit
def _fused_scatter_reduce_forward_kernel(
    inputs_ptr,
    index_ptr,
    out_ptr,
    num_feats,
    num_reductions,
    numel,
    REDUCE0,
    REDUCE1,
    REDUCE2,
    REDUCE3,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    inputs = tl.load(inputs_ptr + offsets, mask=mask)

    index_offsets = offsets // num_feats
    index = tl.load(index_ptr + index_offsets, mask=mask)

    # NOTE Triton does not support for-loops. As such, we cap the maximum
    # number of fused operations to `4` and unroll the loop.
    # TODO (matthias) Try to clean this up.

    if REDUCE0 > 0:
        out_offsets = (num_feats * num_reductions) * index
        out_offsets = out_offsets + (offsets % num_feats)
        if REDUCE0 == 1:  # sum
            tl.atomic_add(out_ptr + out_offsets, inputs, mask=mask)
        elif REDUCE0 == 2:  # mean
            tl.atomic_add(out_ptr + out_offsets, inputs, mask=mask)
        elif REDUCE0 == 3:  # min
            tl.atomic_min(out_ptr + out_offsets, inputs, mask=mask)
        elif REDUCE0 == 4:  # max
            tl.atomic_max(out_ptr + out_offsets, inputs, mask=mask)

    if REDUCE1 > 0:
        out_offsets = (num_feats * num_reductions) * index
        out_offsets = out_offsets + num_feats
        out_offsets = out_offsets + (offsets % num_feats)
        if REDUCE1 == 1:  # sum
            tl.atomic_add(out_ptr + out_offsets, inputs, mask=mask)
        elif REDUCE1 == 2:  # mean
            tl.atomic_add(out_ptr + out_offsets, inputs, mask=mask)
        elif REDUCE1 == 3:  # min
            tl.atomic_min(out_ptr + out_offsets, inputs, mask=mask)
        elif REDUCE1 == 4:  # max
            tl.atomic_max(out_ptr + out_offsets, inputs, mask=mask)

    if REDUCE2 > 0:
        out_offsets = (num_feats * num_reductions) * index
        out_offsets = out_offsets + (2 * num_feats)
        out_offsets = out_offsets + (offsets % num_feats)
        if REDUCE2 == 1:  # sum
            tl.atomic_add(out_ptr + out_offsets, inputs, mask=mask)
        elif REDUCE2 == 2:  # mean
            tl.atomic_add(out_ptr + out_offsets, inputs, mask=mask)
        elif REDUCE2 == 3:  # min
            tl.atomic_min(out_ptr + out_offsets, inputs, mask=mask)
        elif REDUCE2 == 4:  # max
            tl.atomic_max(out_ptr + out_offsets, inputs, mask=mask)

    if REDUCE3 > 0:
        out_offsets = (num_feats * num_reductions) * index
        out_offsets = out_offsets + (3 * num_feats)
        out_offsets = out_offsets + (offsets % num_feats)
        if REDUCE3 == 1:  # sum
            tl.atomic_add(out_ptr + out_offsets, inputs, mask=mask)
        elif REDUCE3 == 2:  # mean
            tl.atomic_add(out_ptr + out_offsets, inputs, mask=mask)
        elif REDUCE3 == 3:  # min
            tl.atomic_min(out_ptr + out_offsets, inputs, mask=mask)
        elif REDUCE3 == 4:  # max
            tl.atomic_max(out_ptr + out_offsets, inputs, mask=mask)


def fused_scatter_reduce_forward(
    inputs: Tensor,
    index: Tensor,
    out: Tensor,
    num_feats: int,
    num_reductions: int,
    reduce_ids: Tuple[int, int, int, int],
) -> None:
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(inputs.numel(), meta['BLOCK_SIZE']),
    )

    _fused_scatter_reduce_forward_kernel[grid](
        inputs,
        index,
        out,
        num_feats,
        num_reductions,
        inputs.numel(),
        reduce_ids[0],
        reduce_ids[1],
        reduce_ids[2],
        reduce_ids[3],
        BLOCK_SIZE=256,
    )
