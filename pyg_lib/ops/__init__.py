from typing import List, Optional, Tuple

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
        input = torch.nested.as_nested_tensor(inputs).contiguous()
        other = torch.nested.as_nested_tensor(others).contiguous()
        if input.dim() == 4 or other.dim() == 4:
            # bmm only works on lists of 2D tensors
            out = torch.matmul(input, other).contiguous()
        else:
            out = torch.bmm(input, other).contiguous()
        outs = list(out.unbind())
    else:
        input_req_grad = any([i.requires_grad for i in inputs])
        other_req_grad = any([i.requires_grad for i in others])
        if input_req_grad or other_req_grad:
            raise ValueError("Autograd is not supported in `grouped_matmul` "
                             "for PyTorch <= 1.13. Please `detach()` your "
                             "input tensors before calling this function.")

        outs = torch.ops.pyg.grouped_matmul(inputs, others)
    if biases is not None:
        for i in range(len(biases)):
            outs[i] = outs[i] + biases[i]
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


def index_sort(
    inputs: Tensor,
    max_value: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Sorts the elements of the :obj:`inputs` tensor in ascending order.
    It is expected that :obj:`inputs` is one-dimensional and that it only
    contains positive integer values. If :obj:`max_value` is given, it can be
    used by the underlying algorithm for better performance.

    .. note::

        This operation is optimized only for tensors associated with the CPU
        device.

    Args:
        inputs (torch.Tensor): A vector with positive integer values.
        max_value (int, optional): The maximum value stored inside
            :obj:`inputs`. This value can be an estimation, but needs to be
            greater than or equal to the real maximum. (default: :obj:`None`)

    Returns:
        Tuple[torch.LongTensor, torch.LongTensor]:
        A tuple containing sorted values and indices of the elements in the
        original :obj:`input` tensor.
    """
    if inputs.is_cuda:
        return torch.sort(inputs)
    return torch.ops.pyg.index_sort(inputs, max_value)


__all__ = [
    'grouped_matmul',
    'segment_matmul',
    'sampled_add',
    'sampled_sub',
    'sampled_mul',
    'sampled_div',
    'index_sort',
    'fused_scatter_reduce',
]
