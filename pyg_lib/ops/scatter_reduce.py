from typing import List

from torch import Tensor


def fused_scatter_reduce(inputs: Tensor, index: Tensor, dim_size: int,
                         reduce: List[str]) -> Tensor:
    # TODO (matthias): Add support for `out`.
    # TODO (matthias): Add backward functionality.
    # TODO (matthias): Add support for inputs.dim() != 2.
    # TODO (matthias): Add support for index.dim() != 1.
    # TODO (matthias) Add support for `dim` argument.
    assert len(reduce) > 1
    assert inputs.is_cuda and index.is_cuda
    assert inputs.dim() == 2 and index.dim() == 1

    out = inputs.new(dim_size, len(reduce) * inputs.size(1))

    return out
