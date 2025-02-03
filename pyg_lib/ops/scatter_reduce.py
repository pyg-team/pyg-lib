import warnings
from typing import List

from torch import Tensor

from pyg_lib._triton import tl, triton

REDUCTIONS = ['sum', 'mean', 'min', 'max']
NUM_REDUCTIONS = len(REDUCTIONS)
NONE = 'none'

OPT_REDUCTIONS = [NONE] + REDUCTIONS


@triton.jit
def _fused_scatter_reduce_forward_kernel(inputs_ptr, index_ptr, out_ptr,
                                         num_feats, num_reductions, numel,
                                         REDUCE0, REDUCE1, REDUCE2, REDUCE3,
                                         BLOCK_SIZE: tl.constexpr):
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
        elif REDUCE2 == 3:  # min
            tl.atomic_min(out_ptr + out_offsets, inputs, mask=mask)
        elif REDUCE3 == 4:  # max
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


def fused_scatter_reduce(
    inputs: Tensor,
    index: Tensor,
    dim_size: int,
    reduce_list: List[str],
) -> Tensor:
    r"""Fuses multiple scatter operations into a single kernel."""
    # TODO (matthias): Add support for `out`.
    # TODO (matthias): Add backward functionality.
    # TODO (matthias): Add support for inputs.dim() != 2.
    # TODO (matthias): Add support for index.dim() != 1.
    # TODO (matthias) Add support for `dim` argument.
    assert inputs.is_floating_point()
    assert inputs.is_cuda and index.is_cuda
    assert inputs.dim() == 2 and index.dim() == 1
    assert inputs.size(0) == index.size(0)
    assert inputs.is_contiguous() and index.is_contiguous()

    num_feats = inputs.size(1)
    num_reductions = len(reduce_list)

    assert isinstance(reduce_list, list) and len(reduce_list) <= NUM_REDUCTIONS

    if len(reduce_list) <= 1:
        warnings.warn(f"It is not recommended to call `fused_scatter_reduce` "
                      f"with a single reduction (got {reduce_list}). Please "
                      f"consider using vanilla `scatter_reduce_` instead.")

    reduce_slice_dict = {
        reduce: slice(i * num_feats, (i + 1) * num_feats)
        for i, reduce in enumerate(reduce_list)
    }
    assert len(reduce_list) == len(reduce_slice_dict)

    out = inputs.new(dim_size, len(reduce_list) * num_feats)

    # Pre-processing: Take care of correct initialization for each reduction:
    for reduce in reduce_list:
        assert reduce in REDUCTIONS
        if reduce == 'min':
            fill_value = float('inf')
        elif reduce == 'max':
            fill_value = float('-inf')
        else:
            fill_value = 0.0
        out[:, reduce_slice_dict[reduce]] = fill_value

    # Fill `reduce_list` with dummy values.
    reduce_list = reduce_list + [NONE] * (NUM_REDUCTIONS - len(reduce_list))

    # TODO (matthias) Do not compute "sum" and "mean" reductions twice.

    grid = lambda meta: (  # noqa: E731
        triton.cdiv(inputs.numel(), meta['BLOCK_SIZE']), )

    _fused_scatter_reduce_forward_kernel[grid](
        inputs,
        index,
        out,
        num_feats,
        num_reductions,
        inputs.numel(),
        OPT_REDUCTIONS.index(reduce_list[0]),
        OPT_REDUCTIONS.index(reduce_list[1]),
        OPT_REDUCTIONS.index(reduce_list[2]),
        OPT_REDUCTIONS.index(reduce_list[3]),
        BLOCK_SIZE=256,
    )

    # Post-processing:
    if 'mean' in reduce_slice_dict:
        degree = inputs.new_zeros(dim_size)
        degree.scatter_add_(0, index, inputs.new_ones(index.numel()))
        degree.clamp_(min=1.0)
        tmp = out[:, reduce_slice_dict['mean']]
        tmp /= degree.view(-1, 1)
    if 'min' in reduce_slice_dict:
        tmp = out[:, reduce_slice_dict['min']]
        tmp[tmp == float('inf')] = 0.
    if 'max' in reduce_slice_dict:
        tmp = out[:, reduce_slice_dict['max']]
        tmp[tmp == float('-inf')] = 0.

    return out
