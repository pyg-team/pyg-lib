from typing import List

from torch import Tensor


def fused_scatter_reduce(inputs: Tensor, index: Tensor, dim_size: int,
                         reduce: List[str]) -> Tensor:
    pass
