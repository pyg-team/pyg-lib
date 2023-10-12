from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

import torch.utils._pytree as pytree


def pytreeify(cls):
    r"""A pytree is Python nested data structure. It is a tree in the sense
    that nodes are Python collections (e.g., list, tuple, dict) and the leaves
    are Python values.
    pytrees are useful for working with nested collections of Tensors. For
    example, one can use `tree_map` to map a function over all Tensors inside
    some nested collection of Tensors and `tree_unflatten` to get a flat list
    of all Tensors inside some nested collection.
    """
    assert issubclass(cls, torch.autograd.Function)

    # The original functions we will replace with flattened versions:
    orig_fw = cls.forward
    orig_bw = cls.backward
    orig_apply = cls.apply

    def new_apply(*inputs):
        flat_inputs, struct = pytree.tree_flatten(inputs)
        out_struct_holder = []
        flat_out = orig_apply(struct, out_struct_holder, *flat_inputs)
        assert len(out_struct_holder) == 1
        return pytree.tree_unflatten(flat_out, out_struct_holder[0])

    def new_forward(ctx, struct, out_struct_holder, *flat_inputs):
        inputs = pytree.tree_unflatten(flat_inputs, struct)

        out = orig_fw(ctx, *inputs)

        flat_out, out_struct = pytree.tree_flatten(out)
        ctx._inp_struct = struct
        ctx._out_struct = out_struct
        out_struct_holder.append(out_struct)
        return tuple(flat_out)

    def new_backward(ctx, *flat_grad_outputs):
        outs_grad = pytree.tree_unflatten(flat_grad_outputs, ctx._out_struct)
        if not isinstance(outs_grad, tuple):
            outs_grad = (outs_grad, )

        grad_inputs = orig_bw(ctx, *outs_grad)

        flat_grad_inputs, grad_inputs_struct = pytree.tree_flatten(grad_inputs)
        return (None, None) + tuple(flat_grad_inputs)

    cls.apply = new_apply
    cls.forward = new_forward
    cls.backward = new_backward

    return cls


@pytreeify
class GroupedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, args: Tuple[Tensor]) -> Tuple[Tensor]:
        ctx.save_for_backward(*args)

        inputs: List[Tensor] = [x for x in args[:int(len(args) / 2)]]
        others: List[Tensor] = [other for other in args[int(len(args) / 2):]]
        outs = torch.ops.pyg.grouped_matmul(inputs, others)

        # NOTE Autograd doesnt set `out[i].requires_grad = True` automatically
        for x, other, out in zip(inputs, others, outs):
            if x.requires_grad or other.requires_grad:
                out.requires_grad = True

        return tuple(outs)

    @staticmethod
    def backward(ctx, *outs_grad: Tuple[Tensor]) -> Tuple[Tensor]:
        args = ctx.saved_tensors
        inputs: List[Tensor] = [x for x in args[:int(len(outs_grad))]]
        others: List[Tensor] = [other for other in args[int(len(outs_grad)):]]

        inputs_grad = []
        if any([x.requires_grad for x in inputs]):
            others = [other.t() for other in others]
            inputs_grad = torch.ops.pyg.grouped_matmul(outs_grad, others)
        else:
            inputs_grad = [None for i in range(len(outs_grad))]

        others_grad = []
        if any([other.requires_grad for other in others]):
            inputs = [x.t() for x in inputs]
            others_grad = torch.ops.pyg.grouped_matmul(inputs, outs_grad)
        else:
            others_grad = [None for i in range(len(outs_grad))]

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
    # Combine inputs into a single tuple for autograd:
    outs = list(GroupedMatmul.apply(tuple(inputs + others)))

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
    if inputs.is_cuda or inputs.is_xpu:
        return torch.sort(inputs)
    return torch.ops.pyg.index_sort(inputs, max_value)


class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        src: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        dim: int = 0,
    ) -> Tensor:
        out = torch.ops.pyg.softmax_forward(src, index, ptr, num_nodes, dim)
        ctx.save_for_backward(out, index, ptr)
        ctx.num_nodes = num_nodes
        ctx.dim = dim

        return out

    @staticmethod
    def backward(ctx, out_grad: Tensor) -> Tuple[Union[Tensor, int]]:
        out, index, ptr = ctx.saved_tensors
        in_grad = torch.ops.pyg.softmax_backward(
            out, out_grad, index, ptr, ctx.num_nodes, ctx.dim
        )

        return in_grad, None, None, None, None


def softmax(
    src: Tensor,
    index: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    dim: int = 0,
) -> Tensor:
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the given dimension :attr:`dim`, based on the indices specified in
    :attr:`index`, and then proceeds to compute the softmax individually for
    each group.

    .. note::

        This operation is currently implemented only for 2D data, where
        segments are created along the first dimension and are defined using
        ptr.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor, optional): The indices of elements for applying the
            softmax. (default: :obj:`None`)
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dim (int, optional): The dimension in which to normalize.
            (default: :obj:`0`)

    :rtype: :class:`Tensor`

    Examples:

        >>> src = torch.randn(4, 4)
        >>> ptr = torch.tensor([0, 4])
        >>> softmax(src, None, ptr)
        tensor([[0.0157, 0.0984, 0.1250, 0.4523],
                [0.1453, 0.2591, 0.5907, 0.2410],
                [0.0598, 0.2923, 0.1206, 0.0921],
                [0.7792, 0.3502, 0.1638, 0.2145]])
    """
    if src.dim() != 2 or not src.is_cpu or ptr is None or dim != 0:
        # currently softmax is implemented for GAT cases:
        # - src is of shape(X, num_heads) and associated with CPU device
        # - ptr is given
        # - dim is 0
        raise NotImplementedError

    return Softmax.apply(src, index, ptr, num_nodes, dim)


__all__ = [
    'grouped_matmul',
    'segment_matmul',
    'sampled_add',
    'sampled_sub',
    'sampled_mul',
    'sampled_div',
    'index_sort',
    'softmax',
]
