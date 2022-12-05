from typing import List

import torch
from torch import Tensor

from .softmax import softmax
from .scatter_reduce import fused_scatter_reduce


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
    major_vers, minor_vers = str(torch.__version__).split('.')[:2]

    if int(major_vers) >= 2 or int(minor_vers) >= 14:
        inputs = torch.nested.as_nested_tensor(inputs).contiguous()
        others = torch.nested.as_nested_tensor(others).contiguous()
        return list(torch.bmm(inputs, others).contiguous().unbind())
    else:
        input_req_grad = any([i.requires_grad for i in inputs])
        other_req_grad = any([i.requires_grad for i in others])
        if input_req_grad or other_req_grad:
            raise ValueError("Autograd is not supported in `grouped_matmul` "
                             "for PyTorch < 1.14. Please `detach()` your "
                             "input tensors before calling this function.")

        return torch.ops.pyg.grouped_matmul(inputs, others)


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
    return torch.ops.pyg.segment_matmul(inputs, ptr, other)


__all__ = [
    'grouped_matmul',
    'segment_matmul',
    'softmax',
    'fused_scatter_reduce',
]
