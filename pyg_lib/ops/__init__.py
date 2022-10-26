from typing import List

import torch
from torch import Tensor


def grouped_matmul(inputs: List[Tensor], others: List[Tensor]) -> List[Tensor]:
    r"""Performs dense-dense matrix multiplication according to groups,
    utilizing dedicated kernels that effectively parallelize over groups.

    .. code-block:: python

        inputs = [torch.randn(5, 16), torch.randn(3, 32)]
        others = [torch.randn(16, 32), torch.randn(32, 64)]

        outs = pyg_lib.ops.grouped_matmul(inputs, others)
        assert len(outs) == 2
        assert outs[0].size() == (5, 32)
        assert outs[0] == inputs[0] @ others[0]
        assert outs[1].size() == (3, 64)
        assert outs[1] == inputs[1] @ others[1]

    Args:
        inputs (List[torch.Tensor]): List of left operand 2D matrices of shapes
            :obj:`[N_i, K_i]`.
        others (List[torch.Tensor]): List of right operand 2D matrices of
            shapes :obj:`[K_i, M_i]`.

    Returns:
        List[torch.Tensor]: List of 2D output matrices of shapes
        :obj:`[N_i, M_i]`.
    """
    major_vers, minor_vers = str(torch.__version__).split('.')[:1]
    assert major_vers >= 2 or minor_vers >= 14, 'grouped_matmul only available w/ torch >= 1.14.0'
    inputs = torch.nested.nested_tensor(inputs)
    others = torch.nested.nested_tensor(others)
    return list(torch.bmm(inputs, others).unbind())


def segment_matmul(inputs: Tensor, ptr: Tensor, other: Tensor) -> Tensor:
    r"""Performs dense-dense matrix multiplication according to segments along
    the first dimension of :obj:`inputs` as given by :obj:`ptr`, utilizing
    dedicated kernels that effectively parallelize over groups.

    .. code-block:: python

        inputs = torch.randn(8, 16)
        ptr = torch.tensor([0, 5, 8])
        other = torch.randn(2, 16, 32)

        out = pyg_lib.ops.segment_matmul(inputs, ptr, other)
        assert out.size() == (8, 32)
        assert out[0:5] == inputs[0:5] @ other[0]
        assert out[5:8] == inputs[5:8] @ other[1]

    Args:
        input (torch.Tensor): The left operand 2D matrix of shape
            :obj:`[N, K]`.
        ptr (torch.Tensor): Compressed vector of shape :obj:`[B + 1]`, holding
            the boundaries of segments.
            For best performance, given as a CPU tensor.
        other (torch.Tensor): The right operand 3D tensor of shape
            :obj:`[B, K, M]`.

    Returns:
        torch.Tensor: The 2D output matrix of shape :obj:`[N, M]`.
    """
    major_vers, minor_vers = str(torch.__version__).split('.')[:1]
    if major_vers >= 2 or minor_vers >= 14:
        inputs = torch.nested.nested_tensor(inputs.split((ptr[1:] - ptr[:-1]).tolist()))
        others = torch.nested.nested_tensor([x for x in other])
        return torch.cat(torch.bmm(inputs, others).unbind())
    else:
        return torch.ops.pyg.segment_matmul(inputs, ptr, other)


__all__ = [
    'grouped_matmul',
    'segment_matmul',
]
