from typing import List, Optional

import torch
from torch import Tensor

from .scatter_reduce import fused_scatter_reduce


def grouped_matmul(inputs: List[Tensor], others: List[Tensor],
                   biases: Optional[List[Tensor]] = None) -> List[Tensor]:
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
        biases (List[torch.Tensor], optional): Optional bias terms to apply for
            each element. (default: :obj:`None`)

    Returns:
        List[torch.Tensor]: List of 2D output matrices of shapes
        :obj:`[N_i, M_i]`.
    """
    major_vers, minor_vers = str(torch.__version__).split('.')[:2]

    if int(major_vers) >= 2 or int(minor_vers) >= 14:
        inputs = torch.nested.as_nested_tensor(inputs).contiguous()
        others = torch.nested.as_nested_tensor(others).contiguous()
        outs = torch.bmm(inputs, others).contiguous()
        if biases is not None:
            for i in range(len(biases)):
                outs[i] += biases[i]
        outs = list(outs.unbind())
    else:
        input_req_grad = any([i.requires_grad for i in inputs])
        other_req_grad = any([i.requires_grad for i in others])
        if input_req_grad or other_req_grad:
            raise ValueError("Autograd is not supported in `grouped_matmul` "
                             "for PyTorch < 1.14. Please `detach()` your "
                             "input tensors before calling this function.")

        outs = torch.ops.pyg.grouped_matmul(inputs, others)
        if biases is not None:
            for i in range(len(biases)):
                outs[i] += biases[i]
    return outs


def segment_matmul(inputs: Tensor, ptr: Tensor, other: Tensor,
                   bias: Optional[Tensor] = None) -> Tensor:
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
        bias (torch.Tensor, optional): Optional bias term of shape
            :obj:`[B, M]` (default: :obj:`None`)

    Returns:
        torch.Tensor: The 2D output matrix of shape :obj:`[N, M]`.
    """
    out = torch.ops.pyg.segment_matmul(inputs, ptr, other)
    if bias is not None:
        for i in range(ptr.numel() - 1):
            out[ptr[i]:ptr[i + 1]] += bias[i]
    return out


def sampled_add(
    left: Tensor,
    right: Tensor,
    left_index: Optional[Tensor] = None,
    right_index: Optional[Tensor] = None,
) -> Tensor:
    r"""Performs a sampled **addition** of :obj:`left` and :obj:`right`
    according to the indices specified in :obj:`left_index` and
    :obj:`right_index`:

    .. math::
        \textrm{out} = \textrm{left}[\textrm{left_index}] +
        \textrm{right}[\textrm{right_index}]

    This operation fuses the indexing and addition operation together, thus
    being more runtime and memory-efficient.

    Args:
        left (torch.Tensor): The left tensor.
        right (torch.Tensor): The right tensor.
        left_index (torch.LongTensor, optional): The values to sample from the
            :obj:`left` tensor. (default: :obj:`None`)
        right_index (torch.LongTensor, optional): The values to sample from the
            :obj:`right` tensor. (default: :obj:`None`)

    Returns:
        torch.Tensor: The output tensor.
    """
    out = torch.ops.pyg.sampled_op(left, right, left_index, right_index, "add")
    return out


def sampled_sub(
    left: Tensor,
    right: Tensor,
    left_index: Optional[Tensor] = None,
    right_index: Optional[Tensor] = None,
) -> Tensor:
    r"""Performs a sampled **subtraction** of :obj:`left` by :obj:`right`
    according to the indices specified in :obj:`left_index` and
    :obj:`right_index`:

    .. math::
        \textrm{out} = \textrm{left}[\textrm{left_index}] -
        \textrm{right}[\textrm{right_index}]

    This operation fuses the indexing and subtraction operation together, thus
    being more runtime and memory-efficient.

    Args:
        left (torch.Tensor): The left tensor.
        right (torch.Tensor): The right tensor.
        left_index (torch.LongTensor, optional): The values to sample from the
            :obj:`left` tensor. (default: :obj:`None`)
        right_index (torch.LongTensor, optional): The values to sample from the
            :obj:`right` tensor. (default: :obj:`None`)

    Returns:
        torch.Tensor: The output tensor.
    """
    out = torch.ops.pyg.sampled_op(left, right, left_index, right_index, "sub")
    return out


def sampled_mul(
    left: Tensor,
    right: Tensor,
    left_index: Optional[Tensor] = None,
    right_index: Optional[Tensor] = None,
) -> Tensor:
    r"""Performs a sampled **multiplication** of :obj:`left` and :obj:`right`
    according to the indices specified in :obj:`left_index` and
    :obj:`right_index`:

    .. math::
        \textrm{out} = \textrm{left}[\textrm{left_index}] *
        \textrm{right}[\textrm{right_index}]

    This operation fuses the indexing and multiplication operation together,
    thus being more runtime and memory-efficient.

    Args:
        left (torch.Tensor): The left tensor.
        right (torch.Tensor): The right tensor.
        left_index (torch.LongTensor, optional): The values to sample from the
            :obj:`left` tensor. (default: :obj:`None`)
        right_index (torch.LongTensor, optional): The values to sample from the
            :obj:`right` tensor. (default: :obj:`None`)

    Returns:
        torch.Tensor: The output tensor.
    """
    out = torch.ops.pyg.sampled_op(left, right, left_index, right_index, "mul")
    return out


def sampled_div(
    left: Tensor,
    right: Tensor,
    left_index: Optional[Tensor] = None,
    right_index: Optional[Tensor] = None,
) -> Tensor:
    r"""Performs a sampled **division** of :obj:`left` by :obj:`right`
    according to the indices specified in :obj:`left_index` and
    :obj:`right_index`:

    .. math::
        \textrm{out} = \textrm{left}[\textrm{left_index}] /
        \textrm{right}[\textrm{right_index}]

    This operation fuses the indexing and division operation together, thus
    being more runtime and memory-efficient.

    Args:
        left (torch.Tensor): The left tensor.
        right (torch.Tensor): The right tensor.
        left_index (torch.LongTensor, optional): The values to sample from the
            :obj:`left` tensor. (default: :obj:`None`)
        right_index (torch.LongTensor, optional): The values to sample from the
            :obj:`right` tensor. (default: :obj:`None`)

    Returns:
        torch.Tensor: The output tensor.
    """
    out = torch.ops.pyg.sampled_op(left, right, left_index, right_index, "div")
    return out


__all__ = [
    'grouped_matmul',
    'segment_matmul',
    'sampled_add',
    'sampled_sub',
    'sampled_mul',
    'sampled_div',
    'fused_scatter_reduce',
]
