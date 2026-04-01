from typing import List, Optional, Tuple

import torch
import torch.utils._pytree as pytree
from torch import Tensor


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
            outs_grad = (outs_grad,)

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

        inputs: List[Tensor] = [x for x in args[: int(len(args) / 2)]]
        others: List[Tensor] = [other for other in args[int(len(args) / 2) :]]
        outs = torch.ops.pyg.grouped_matmul(inputs, others)

        # NOTE Autograd doesnt set `out[i].requires_grad = True` automatically
        for x, other, out in zip(inputs, others, outs):
            if x.requires_grad or other.requires_grad:
                out.requires_grad = True

        return tuple(outs)

    @staticmethod
    def backward(ctx, *outs_grad: Tuple[Tensor]) -> Tuple[Tensor]:
        args = ctx.saved_tensors
        inputs: List[Tensor] = [x for x in args[: int(len(outs_grad))]]
        others: List[Tensor] = [other for other in args[int(len(outs_grad)) :]]

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
            out[ptr[i] : ptr[i + 1]] += bias[i]
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
    out = torch.ops.pyg.sampled_op(left, right, left_index, right_index, 'add')
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
    out = torch.ops.pyg.sampled_op(left, right, left_index, right_index, 'sub')
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
    out = torch.ops.pyg.sampled_op(left, right, left_index, right_index, 'mul')
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
    out = torch.ops.pyg.sampled_op(left, right, left_index, right_index, 'div')
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


def spline_basis(
    pseudo: Tensor,
    kernel_size: Tensor,
    is_open_spline: Tensor,
    degree: int = 1,
) -> Tuple[Tensor, Tensor]:
    r"""Computes the B-spline basis functions.

    Args:
        pseudo: Pseudo-coordinates of shape :obj:`[E, D]`.
        kernel_size: Kernel size in each dimension of shape :obj:`[D]`.
        is_open_spline: Whether to use open B-splines of shape :obj:`[D]`.
        degree: B-spline degree (1, 2, or 3).

    Returns:
        Basis values of shape :obj:`[E, S]` and weight indices of shape
        :obj:`[E, S]`.
    """
    return torch.ops.pyg.spline_basis(
        pseudo,
        kernel_size,
        is_open_spline,
        degree,
    )


def spline_weighting(
    x: Tensor,
    weight: Tensor,
    basis: Tensor,
    weight_index: Tensor,
) -> Tensor:
    r"""Computes the spline weighting of input features.

    Args:
        x: Input features of shape :obj:`[E, M_in]`.
        weight: Weight tensor of shape :obj:`[K, M_in, M_out]`.
        basis: B-spline basis values of shape :obj:`[E, S]`.
        weight_index: Weight indices of shape :obj:`[E, S]`.

    Returns:
        Output features of shape :obj:`[E, M_out]`.
    """
    return torch.ops.pyg.spline_weighting(x, weight, basis, weight_index)


def spline_conv(
    x: Tensor,
    edge_index: Tensor,
    pseudo: Tensor,
    weight: Tensor,
    kernel_size: Tensor,
    is_open_spline: Tensor,
    degree: int = 1,
    norm: bool = True,
    root_weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
) -> Tensor:
    r"""Performs the spline-based convolution operator, combining
    :func:`spline_basis` and :func:`spline_weighting` with message
    aggregation, as described in the `"SplineCNN: Fast Geometric Deep Learning
    with Continuous B-Spline Kernels"
    <https://arxiv.org/abs/1711.08920>`_ paper.

    Args:
        x: Node feature matrix of shape :obj:`[N, M_in]`.
        edge_index: Edge indices of shape :obj:`[2, E]`.
        pseudo: Edge pseudo-coordinates of shape :obj:`[E, D]` with values
            in :obj:`[0, 1]`.
        weight: Trainable weight tensor of shape :obj:`[K, M_in, M_out]`.
        kernel_size: Kernel size in each dimension of shape :obj:`[D]`.
        is_open_spline: Whether to use open B-splines of shape :obj:`[D]`.
        degree: B-spline degree (1, 2, or 3).
        norm: If :obj:`True`, normalizes the output by the node degree.
        root_weight: Optional root weight matrix of shape
            :obj:`[M_in, M_out]` for self-loop features.
        bias: Optional bias vector of shape :obj:`[M_out]`.

    Returns:
        Node output features of shape :obj:`[N, M_out]`.
    """
    N = x.size(0)
    row, col = edge_index[0], edge_index[1]

    basis, weight_index = spline_basis(
        pseudo,
        kernel_size,
        is_open_spline,
        degree,
    )
    out = spline_weighting(x[col], weight, basis, weight_index)

    # Aggregate by target node via scatter_add:
    result = x.new_zeros(N, out.size(1))
    result.scatter_add_(0, row.unsqueeze(1).expand_as(out), out)

    if norm:
        deg = x.new_zeros(N)
        deg.scatter_add_(0, row, x.new_ones(row.size(0)))
        deg = deg.clamp(min=1).unsqueeze(1)
        result = result / deg

    if root_weight is not None:
        result = result + x @ root_weight

    if bias is not None:
        result = result + bias

    return result


def grid_cluster(
    pos: Tensor,
    size: Tensor,
    start: Optional[Tensor] = None,
    end: Optional[Tensor] = None,
) -> Tensor:
    r"""Clusters all points in :obj:`pos` into voxels of size :obj:`size`.

    Each point is assigned a cluster index based on which voxel it falls into.
    The voxel grid is defined by the :obj:`size` parameter and optionally
    bounded by :obj:`start` and :obj:`end`.

    Args:
        pos: Point positions of shape :obj:`[N, D]`.
        size: Voxel size in each dimension of shape :obj:`[D]`.
        start: Start of the voxel grid in each dimension of shape :obj:`[D]`.
            If :obj:`None`, uses the minimum of :obj:`pos`.
        end: End of the voxel grid in each dimension of shape :obj:`[D]`.
            If :obj:`None`, uses the maximum of :obj:`pos`.

    Returns:
        Cluster index for each point of shape :obj:`[N]`.
    """
    return torch.ops.pyg.grid_cluster(pos, size, start, end)


def fps(
    src: Tensor,
    ptr: Tensor,
    ratio: float = 0.5,
    random_start: bool = True,
) -> Tensor:
    r"""Performs greedy farthest point sampling.

    Starting from a random point (or the first point), iteratively selects
    the point that is farthest from the already selected set.

    Args:
        src: Point positions of shape :obj:`[N, D]`.
        ptr: Batch boundaries as a CSR pointer of shape :obj:`[B + 1]`.
        ratio: Fraction of points to sample from each batch (in :obj:`(0, 1]`).
        random_start: If :obj:`True`, starts from a random point.

    Returns:
        Indices of the sampled points of shape :obj:`[M]`.
    """
    return torch.ops.pyg.fps(src, ptr, ratio, random_start)


def knn(
    x: Tensor,
    y: Tensor,
    k: int = 1,
    ptr_x: Optional[Tensor] = None,
    ptr_y: Optional[Tensor] = None,
    cosine: bool = False,
    num_workers: int = 1,
) -> Tensor:
    r"""Finds for each element in :obj:`y` the :obj:`k` nearest points in
    :obj:`x`.

    Args:
        x: Reference points of shape :obj:`[N, D]`.
        y: Query points of shape :obj:`[M, D]`.
        k: Number of nearest neighbors.
        ptr_x: Batch boundaries for :obj:`x` as a CSR pointer.
        ptr_y: Batch boundaries for :obj:`y` as a CSR pointer.
        cosine: If :obj:`True`, uses cosine distance (CUDA only).
        num_workers: Number of workers (unused, for API compat).

    Returns:
        Edge indices of shape :obj:`[2, M*k]` where row 0 is query indices
        and row 1 is reference indices.
    """
    return torch.ops.pyg.knn(x, y, ptr_x, ptr_y, k, cosine, num_workers)


def knn_graph(
    x: Tensor,
    k: int,
    ptr: Optional[Tensor] = None,
    loop: bool = False,
    flow: str = 'source_to_target',
    cosine: bool = False,
    num_workers: int = 1,
) -> Tensor:
    r"""Constructs a k-nearest neighbor graph for the given node features.

    Args:
        x: Node feature matrix of shape :obj:`[N, D]`.
        k: Number of nearest neighbors.
        ptr: Batch boundaries as a CSR pointer of shape :obj:`[B + 1]`.
        loop: If :obj:`True`, includes self-loops in the output.
        flow: The direction of edges, either :obj:`"source_to_target"` or
            :obj:`"target_to_source"`.
        cosine: If :obj:`True`, uses cosine distance (CUDA only).
        num_workers: Number of workers (unused, for API compat).

    Returns:
        Edge indices of shape :obj:`[2, E]`.
    """
    actual_k = k if loop else k + 1
    edge_index = knn(
        x,
        x,
        actual_k,
        ptr_x=ptr,
        ptr_y=ptr,
        cosine=cosine,
        num_workers=num_workers,
    )

    if not loop:
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]

    if flow == 'target_to_source':
        edge_index = edge_index.flip(0)

    return edge_index


def radius(
    x: Tensor,
    y: Tensor,
    r: float = 1.0,
    ptr_x: Optional[Tensor] = None,
    ptr_y: Optional[Tensor] = None,
    max_num_neighbors: int = 32,
    num_workers: int = 1,
    ignore_same_index: bool = False,
) -> Tensor:
    r"""Finds all points in :obj:`x` within distance :obj:`r` of points in
    :obj:`y`.

    Args:
        x: Reference points of shape :obj:`[N, D]`.
        y: Query points of shape :obj:`[M, D]`.
        r: Radius.
        ptr_x: Batch boundaries for :obj:`x` as a CSR pointer.
        ptr_y: Batch boundaries for :obj:`y` as a CSR pointer.
        max_num_neighbors: Maximum number of neighbors per query point.
        num_workers: Number of workers (unused, for API compat).
        ignore_same_index: If :obj:`True`, ignores pairs with same index.

    Returns:
        Edge indices of shape :obj:`[2, E]` where row 0 is query indices
        and row 1 is reference indices.
    """
    return torch.ops.pyg.radius(
        x,
        y,
        ptr_x,
        ptr_y,
        r,
        max_num_neighbors,
        num_workers,
        ignore_same_index,
    )


def radius_graph(
    x: Tensor,
    r: float,
    ptr: Optional[Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = 'source_to_target',
    num_workers: int = 1,
) -> Tensor:
    r"""Constructs a radius graph for the given node features.

    Finds all pairs of nodes within distance :obj:`r` and returns their
    edge indices.

    Args:
        x: Node feature matrix of shape :obj:`[N, D]`.
        r: Radius.
        ptr: Batch boundaries as a CSR pointer of shape :obj:`[B + 1]`.
        loop: If :obj:`True`, includes self-loops in the output.
        max_num_neighbors: Maximum number of neighbors per node.
        flow: The direction of edges, either :obj:`"source_to_target"` or
            :obj:`"target_to_source"`.
        num_workers: Number of workers (unused, for API compat).

    Returns:
        Edge indices of shape :obj:`[2, E]`.
    """
    edge_index = radius(
        x,
        x,
        r,
        ptr_x=ptr,
        ptr_y=ptr,
        max_num_neighbors=max_num_neighbors,
        num_workers=num_workers,
        ignore_same_index=not loop,
    )

    if flow == 'target_to_source':
        edge_index = edge_index.flip(0)

    return edge_index


def nearest(
    x: Tensor,
    y: Tensor,
    ptr_x: Optional[Tensor] = None,
    ptr_y: Optional[Tensor] = None,
) -> Tensor:
    r"""Finds the nearest point in :obj:`y` for each point in :obj:`x`.

    Args:
        x: Query points of shape :obj:`[N, D]`.
        y: Reference points of shape :obj:`[M, D]`.
        ptr_x: Batch boundaries for :obj:`x` as a CSR pointer.
        ptr_y: Batch boundaries for :obj:`y` as a CSR pointer.

    Returns:
        Index tensor of shape :obj:`[N]` with the index of the nearest
        point in :obj:`y` for each point in :obj:`x`.
    """
    return torch.ops.pyg.nearest(x, y, ptr_x, ptr_y)


def graclus_cluster(
    rowptr: Tensor,
    col: Tensor,
    weight: Optional[Tensor] = None,
) -> Tensor:
    r"""Computes a greedy graph clustering via the Graclus algorithm.

    Nodes are matched greedily in random order. The cluster ID for a
    matched pair (u, v) is :obj:`min(u, v)`. Unmatched nodes are assigned
    their own index as cluster ID.

    Args:
        rowptr: CSR row pointer of shape :obj:`[N + 1]`.
        col: Column indices of shape :obj:`[E]`.
        weight: Optional edge weights of shape :obj:`[E]`.

    Returns:
        Cluster assignment of shape :obj:`[N]`.
    """
    return torch.ops.pyg.graclus_cluster(rowptr, col, weight)


def edge_sample(
    start: Tensor,
    rowptr: Tensor,
    count: int = 0,
    factor: float = 1.0,
) -> Tensor:
    r"""Samples edges incident to the given start nodes.

    For each start node, samples up to :obj:`count` edges. If
    :obj:`count < 1`, samples :obj:`ceil(factor * degree)` edges instead.

    Args:
        start: Start node indices of shape :obj:`[S]`.
        rowptr: CSR row pointer of shape :obj:`[N + 1]`.
        count: Fixed number of edges to sample per node. If :obj:`< 1`,
            uses :obj:`factor` instead.
        factor: Fraction of edges to sample when :obj:`count < 1`.

    Returns:
        Sampled edge indices (into the edge list).
    """
    return torch.ops.pyg.edge_sample(start, rowptr, count, factor)


__all__ = [
    'grouped_matmul',
    'segment_matmul',
    'sampled_add',
    'sampled_sub',
    'sampled_mul',
    'sampled_div',
    'index_sort',
    'softmax_csr',
    'spline_basis',
    'spline_weighting',
    'spline_conv',
    'grid_cluster',
    'fps',
    'knn',
    'knn_graph',
    'radius',
    'radius_graph',
    'nearest',
    'graclus_cluster',
    'edge_sample',
]
