from typing import List, Optional, Tuple

import torch
import torch.utils._pytree as pytree
from torch import Tensor


def _broadcast_scatter(src: Tensor, other: Tensor, dim: int) -> Tensor:
    r"""Broadcasts scatter index tensor to match dimensions of source tensor.
    
    This utility function ensures that the index tensor has the same shape
    as the source tensor along all dimensions except the scatter dimension.
    """
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def _pytreeify(cls):
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


@_pytreeify
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


def grouped_matmul(
    inputs: List[Tensor],
    others: List[Tensor],
    biases: Optional[List[Tensor]] = None,
) -> List[Tensor]:
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
        inputs: List of left operand 2D matrices of shapes :obj:`[N_i, K_i]`.
        others: List of right operand 2D matrices of shapes :obj:`[K_i, M_i]`.
        biases: Optional bias terms to apply for each element.

    Returns:
        List of 2D output matrices of shapes :obj:`[N_i, M_i]`.
    """
    # Combine inputs into a single tuple for autograd:
    outs = list(GroupedMatmul.apply(tuple(inputs + others)))

    if biases is not None:
        for i in range(len(biases)):
            outs[i] = outs[i] + biases[i]

    return outs


def segment_matmul(
    inputs: Tensor,
    ptr: Tensor,
    other: Tensor,
    bias: Optional[Tensor] = None,
) -> Tensor:
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
        inputs: The left operand 2D matrix of shape :obj:`[N, K]`.
        ptr: Compressed vector of shape :obj:`[B + 1]`, holding the boundaries
            of segments. For best performance, given as a CPU tensor.
        other: The right operand 3D tensor of shape :obj:`[B, K, M]`.
        bias: The bias term of shape :obj:`[B, M]`.

    Returns:
        The 2D output matrix of shape :obj:`[N, M]`.
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
    :obj:`right_index`.

    .. math::
        \textrm{out} = \textrm{left}[\textrm{left_index}] +
        \textrm{right}[\textrm{right_index}]

    This operation fuses the indexing and addition operation together, thus
    being more runtime and memory-efficient.

    Args:
        left: The left tensor.
        right: The right tensor.
        left_index: The values to sample from the :obj:`left` tensor.
        right_index: The values to sample from the :obj:`right` tensor.

    Returns:
        The output tensor.
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
    :obj:`right_index`.

    .. math::
        \textrm{out} = \textrm{left}[\textrm{left_index}] -
        \textrm{right}[\textrm{right_index}]

    This operation fuses the indexing and subtraction operation together, thus
    being more runtime and memory-efficient.

    Args:
        left: The left tensor.
        right: The right tensor.
        left_index: The values to sample from the :obj:`left` tensor.
        right_index: The values to sample from the :obj:`right` tensor.

    Returns:
        The output tensor.
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
    :obj:`right_index`.

    .. math::
        \textrm{out} = \textrm{left}[\textrm{left_index}] *
        \textrm{right}[\textrm{right_index}]

    This operation fuses the indexing and multiplication operation together,
    thus being more runtime and memory-efficient.

    Args:
        left: The left tensor.
        right: The right tensor.
        left_index: The values to sample from the :obj:`left` tensor.
        right_index: The values to sample from the :obj:`right` tensor.

    Returns:
        The output tensor.
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
    :obj:`right_index`.

    .. math::
        \textrm{out} = \textrm{left}[\textrm{left_index}] /
        \textrm{right}[\textrm{right_index}]

    This operation fuses the indexing and division operation together, thus
    being more runtime and memory-efficient.

    Args:
        left: The left tensor.
        right: The right tensor.
        left_index: The values to sample from the :obj:`left` tensor.
        right_index: The values to sample from the :obj:`right` tensor.

    Returns:
        The output tensor.
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
        inputs: A vector with positive integer values.
        max_value: The maximum value stored inside :obj:`inputs`. This value
            can be an estimation, but needs to be greater than or equal to the
            real maximum.

    Returns:
        A tuple containing sorted values and indices of the elements in the
        original :obj:`input` tensor.
    """
    if not inputs.is_cpu:
        return torch.sort(inputs)
    return torch.ops.pyg.index_sort(inputs, max_value)


def softmax_csr(
    src: Tensor,
    ptr: Tensor,
    dim: int = 0,
) -> Tensor:
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the given dimension :attr:`dim`, based on the indices specified via
    :attr:`ptr`, and then proceeds to compute the softmax individually for
    each group.

    Examples:
        >>> src = torch.randn(4, 4)
        >>> ptr = torch.tensor([0, 4])
        >>> softmax(src, ptr)
        tensor([[0.0157, 0.0984, 0.1250, 0.4523],
                [0.1453, 0.2591, 0.5907, 0.2410],
                [0.0598, 0.2923, 0.1206, 0.0921],
                [0.7792, 0.3502, 0.1638, 0.2145]])

    Args:
        src: The source tensor.
        ptr: Groups defined by CSR representation.
        dim: The dimension in which to normalize.
    """
    dim = dim + src.dim() if dim < 0 else dim
    return torch.ops.pyg.softmax_csr(src, ptr, dim)


def scatter_add(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tensor:
    r"""Sums all values from the :obj:`src` tensor into :obj:`out` at the indices
    specified in the :obj:`index` tensor along a given axis :obj:`dim`.
    
    This operation uses optimized CUDA kernels for enhanced performance compared
    to PyTorch's built-in scatter_add_ operation. For each value in :obj:`src`, 
    its output index is specified by its index in :obj:`src` for dimensions 
    outside of :obj:`dim` and by the corresponding value in :obj:`index` for 
    dimension :obj:`dim`.

    This operation is equivalent to::

        out[index[i]][...] += src[i][...]

    but allows for efficient batched operations and broadcasting.

    .. code-block:: python

        src = torch.tensor([1, 3, 2, 4, 5, 6])
        index = torch.tensor([0, 1, 0, 1, 1, 3])
        out = pyg_lib.ops.scatter_add(src, index)
        # Result: [3, 12, 0, 6]

        # 2D example
        src = torch.tensor([[1, 2], [5, 6], [3, 4], [7, 8], [9, 10], [11, 12]])
        index = torch.tensor([0, 1, 0, 1, 1, 3])
        out = pyg_lib.ops.scatter_add(src, index, dim=0)
        # Result: [[4, 6], [21, 24], [0, 0], [11, 12]]

    Args:
        src: The source tensor to scatter from.
        index: The indices of elements to scatter.
        dim: The axis along which to index. Default: :obj:`-1`.
        out: The destination tensor. If not provided, a new tensor is created.
        dim_size: If :obj:`out` is not given, automatically create output with
            size :obj:`dim_size` at dimension :obj:`dim`. If :obj:`dim_size`
            is not given, a minimal sized output tensor is returned.

    Returns:
        The output tensor.
    """
    return torch.ops.pyg.scatter_add(src, index, dim, out, dim_size)


def scatter_mean(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tensor:
    r"""Computes the mean of all values from the :obj:`src` tensor into :obj:`out`
    at the indices specified in the :obj:`index` tensor along a given axis :obj:`dim`.

    This operation uses optimized CUDA kernels for enhanced performance. It first 
    sums values using optimized scatter_add, then divides by the number of 
    contributions to each output element.

    .. code-block:: python

        src = torch.tensor([1.0, 3.0, 2.0, 4.0, 5.0, 6.0])
        index = torch.tensor([0, 1, 0, 1, 1, 3])
        out = pyg_lib.ops.scatter_mean(src, index)
        # Result: [1.5, 4.0, 0.0, 6.0]

    Args:
        src: The source tensor to scatter from.
        index: The indices of elements to scatter.
        dim: The axis along which to index. Default: :obj:`-1`.
        out: The destination tensor. If not provided, a new tensor is created.
        dim_size: If :obj:`out` is not given, automatically create output with
            size :obj:`dim_size` at dimension :obj:`dim`. If :obj:`dim_size`
            is not given, a minimal sized output tensor is returned.

    Returns:
        The output tensor.
    """
    return torch.ops.pyg.scatter_mean(src, index, dim, out, dim_size)


__all__ = [
    'grouped_matmul',
    'segment_matmul',
    'sampled_add',
    'sampled_sub',
    'sampled_mul',
    'sampled_div',
    'index_sort',
    'scatter_add',
    'scatter_mean',
    'softmax_csr',
]
