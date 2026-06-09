import warnings
from typing import List

from torch import Tensor

REDUCTIONS = ['sum', 'mean', 'min', 'max']
NUM_REDUCTIONS = len(REDUCTIONS)
NONE = 'none'

OPT_REDUCTIONS = [NONE] + REDUCTIONS


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
        warnings.warn(
            f'It is not recommended to call `fused_scatter_reduce` '
            f'with a single reduction (got {reduce_list}). Please '
            f'consider using vanilla `scatter_reduce_` instead.',
        )

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

    from pyg_lib.ops._scatter_reduce_triton import fused_scatter_reduce_forward

    fused_scatter_reduce_forward(
        inputs,
        index,
        out,
        num_feats,
        num_reductions,
        (
            OPT_REDUCTIONS.index(reduce_list[0]),
            OPT_REDUCTIONS.index(reduce_list[1]),
            OPT_REDUCTIONS.index(reduce_list[2]),
            OPT_REDUCTIONS.index(reduce_list[3]),
        ),
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
        tmp[tmp == float('inf')] = 0.0
    if 'max' in reduce_slice_dict:
        tmp = out[:, reduce_slice_dict['max']]
        tmp[tmp == float('-inf')] = 0.0

    return out
