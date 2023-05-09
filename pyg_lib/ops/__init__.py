from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.autograd import Function

from .scatter_reduce import fused_scatter_reduce
import torch.utils._pytree as pytree

# Basically wraps things in and out before passing
# it to the real function that the user defined.
# def pytreeify(cls):
#     assert issubclass(cls, Function)

#     orig_fw = cls.forward
#     orig_bw = cls.backward
#     orig_apply = cls.apply

#     def new_apply(*inp):
#         flat_inp, struct = pytree.tree_flatten(inp)
#         out_struct_holder = []
#         flat_out = orig_apply(struct, out_struct_holder, *flat_inp)
#         assert len(out_struct_holder) == 1
#         return pytree.tree_unflatten(flat_out, out_struct_holder[0])

#     def new_forward(ctx, struct, out_struct_holder, *flat_inp):
#         inp = pytree.tree_unflatten(flat_inp, struct)
#         out = orig_fw(ctx, *inp)
#         flat_out, out_struct = pytree.tree_flatten(out)
#         ctx._inp_struct = struct
#         ctx._out_struct = out_struct
#         out_struct_holder.append(out_struct)
#         return tuple(flat_out)

#     def new_backward(ctx, *flat_grad_outputs):
#         grad_outputs = pytree.tree_unflatten(flat_grad_outputs,
#                                              ctx._out_struct)
#         if not isinstance(grad_outputs, tuple):
#             grad_outputs = (grad_outputs, )
#         grad_inputs = orig_bw(ctx, *grad_outputs)
#         flat_grad_inputs, grad_inputs_struct = pytree.tree_flatten(grad_inputs)
#         if grad_inputs_struct != ctx._inp_struct:
#             raise RuntimeError(
#                 "The backward generated an arg structure that doesn't "
#                 "match the forward's input.")
#         return (None, None) + tuple(flat_grad_inputs)
#     cls.apply = new_apply
#     cls.forward = new_forward
#     cls.backward = new_backward
#     return cls


# # @pytreeify
class GroupedMatmul(Function):
    @staticmethod
    def forward(ctx, *inputs_and_others):
        ctx.save_for_backward(*(inputs_and_others))
        inputs = list(inputs_and_others[:int(len(inputs_and_others) / 2)])
        others = list(inputs_and_others[int(len(inputs_and_others) / 2):])
        outs = torch.ops.pyg.grouped_matmul(inputs, others)

        # # NOTE Autograd doesnt set out[i].requires_grad = True automatically
        for i in range(len(outs)):
            outs[i].requires_grad = True

        return tuple(outs)

    @staticmethod
    def backward(ctx, *outs_grad):
        inputs_and_others = list(ctx.saved_tensors)
        inputs = inputs_and_others[:int(len(outs_grad))]
        others = inputs_and_others[int(len(outs_grad)):]
        # explicit typing needed
        outs_grad: List[Tensor] = list(outs_grad)
        inputs_grad = []
        if all([x.requires_grad for x in inputs]):
            for i in range(len(others)):
                others[i] = others[i].t()
            inputs_grad = torch.ops.pyg.grouped_matmul(outs_grad, others)

        others_grad = []
        if all([other.requires_grad for other in others]):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].t()
            others_grad = torch.ops.pyg.grouped_matmul(inputs, outs_grad)

        return tuple(inputs_grad + others_grad)


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
    outs = list(GroupedMatmul.apply(inputs + others))
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
