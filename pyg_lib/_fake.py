r"""Fake kernels for all :obj:`pyg::*` operators.

A fake kernel describes the outputs of an operator (shape, data type and
device) without computing them. :pytorch:`PyTorch` requires it in order to
trace an operator with data-less tensors, which is how :obj:`torch.export` and
:obj:`torch.compile` capture a model.

Each fake kernel has the same signature as the operator schema defined in C++.
"""

from typing import List, Optional, Tuple

import torch
from torch import Tensor


def new_dynamic_size() -> torch.SymInt:
    r"""Returns a size that depends on the data, i.e. is unknown in advance."""
    return torch.library.get_ctx().new_dynamic_size()


def resize(src: Tensor, dim: int, size: int) -> Tensor:
    r"""Returns a tensor like :obj:`src` of the given size in :obj:`dim`."""
    sizes = list(src.shape)
    sizes[dim] = size
    return src.new_empty(sizes)


def check_out(out: Optional[Tensor]) -> None:
    # Writing into `out` cannot be traced since the operator schemas do not
    # declare that it is modified in-place:
    if out is not None:
        raise NotImplementedError("Cannot trace the 'out' argument")


# Reductions ##################################################################


@torch.library.register_fake('pyg::scatter_sum')
@torch.library.register_fake('pyg::scatter_mul')
@torch.library.register_fake('pyg::scatter_mean')
def scatter(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tensor:
    check_out(out)
    # Without `dim_size`, the output size is given by `index.max() + 1`:
    size = new_dynamic_size() if dim_size is None else dim_size
    return resize(src, dim, size)


@torch.library.register_fake('pyg::scatter_min')
@torch.library.register_fake('pyg::scatter_max')
def scatter_arg(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    out = scatter(src, index, dim, out, dim_size)
    return out, out.new_empty(out.shape, dtype=torch.long)


@torch.library.register_fake('pyg::segment_sum_coo')
@torch.library.register_fake('pyg::segment_mean_coo')
def segment_coo(
    src: Tensor,
    index: Tensor,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tensor:
    return scatter(src, index, index.dim() - 1, out, dim_size)


@torch.library.register_fake('pyg::segment_min_coo')
@torch.library.register_fake('pyg::segment_max_coo')
def segment_arg_coo(
    src: Tensor,
    index: Tensor,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    out = segment_coo(src, index, out, dim_size)
    return out, out.new_empty(out.shape, dtype=torch.long)


@torch.library.register_fake('pyg::segment_sum_csr')
@torch.library.register_fake('pyg::segment_mean_csr')
def segment_csr(
    src: Tensor,
    indptr: Tensor,
    out: Optional[Tensor] = None,
) -> Tensor:
    check_out(out)
    size = torch.sym_max(indptr.size(-1) - 1, 0)
    return resize(src, indptr.dim() - 1, size)


@torch.library.register_fake('pyg::segment_min_csr')
@torch.library.register_fake('pyg::segment_max_csr')
def segment_arg_csr(
    src: Tensor,
    indptr: Tensor,
    out: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    out = segment_csr(src, indptr, out)
    return out, out.new_empty(out.shape, dtype=torch.long)


@torch.library.register_fake('pyg::gather_coo')
def gather_coo(
    src: Tensor,
    index: Tensor,
    out: Optional[Tensor] = None,
) -> Tensor:
    check_out(out)
    return resize(src, index.dim() - 1, index.size(-1))


@torch.library.register_fake('pyg::gather_csr')
def gather_csr(
    src: Tensor,
    indptr: Tensor,
    out: Optional[Tensor] = None,
) -> Tensor:
    check_out(out)
    # The output size is given by the last element in `indptr`:
    return resize(src, indptr.dim() - 1, new_dynamic_size())


@torch.library.register_fake('pyg::softmax_csr')
def softmax_csr(src: Tensor, ptr: Tensor, dim: int = 0) -> Tensor:
    return torch.empty_like(src)


@torch.library.register_fake('pyg::softmax_csr_backward')
def softmax_csr_backward(
    out: Tensor,
    out_grad: Tensor,
    ptr: Tensor,
    dim: int = 0,
) -> Tensor:
    return torch.empty_like(out)


# Matrix multiplication #######################################################


@torch.library.register_fake('pyg::grouped_matmul')
def grouped_matmul(input: List[Tensor], other: List[Tensor]) -> List[Tensor]:
    return [x.new_empty(x.size(0), y.size(-1)) for x, y in zip(input, other)]


@torch.library.register_fake('pyg::segment_matmul')
def segment_matmul(input: Tensor, ptr: Tensor, other: Tensor) -> Tensor:
    return input.new_empty(input.size(0), other.size(-1))


@torch.library.register_fake('pyg::sampled_op')
def sampled_op(
    left: Tensor,
    right: Tensor,
    left_index: Optional[Tensor],
    right_index: Optional[Tensor],
    op: str,
) -> Tensor:
    if left_index is not None:
        left = left.index_select(0, left_index)
    if right_index is not None:
        right = right.index_select(0, right_index)
    out = getattr(torch, op)(left, right)
    # The CUDA kernel does not perform type promotion:
    return out.to(left.dtype) if out.is_cuda else out


# Spline ######################################################################


@torch.library.register_fake('pyg::spline_basis')
def spline_basis(
    pseudo: Tensor,
    kernel_size: Tensor,
    is_open_spline: Tensor,
    degree: int = 1,
) -> Tuple[Tensor, Tensor]:
    size = (pseudo.size(0), (degree + 1) ** pseudo.size(1))
    return pseudo.new_empty(size), kernel_size.new_empty(size)


@torch.library.register_fake('pyg::spline_basis_backward')
def spline_basis_backward(
    grad_basis: Tensor,
    pseudo: Tensor,
    kernel_size: Tensor,
    is_open_spline: Tensor,
    degree: int = 1,
) -> Tensor:
    return torch.empty_like(pseudo)


@torch.library.register_fake('pyg::spline_weighting')
def spline_weighting(
    x: Tensor,
    weight: Tensor,
    basis: Tensor,
    weight_index: Tensor,
) -> Tensor:
    return x.new_empty(x.size(0), weight.size(2))


@torch.library.register_fake('pyg::spline_weighting_backward_x')
def spline_weighting_backward_x(
    grad_out: Tensor,
    weight: Tensor,
    basis: Tensor,
    weight_index: Tensor,
) -> Tensor:
    return grad_out.new_empty(grad_out.size(0), weight.size(1))


@torch.library.register_fake('pyg::spline_weighting_backward_weight')
def spline_weighting_backward_weight(
    grad_out: Tensor,
    x: Tensor,
    basis: Tensor,
    weight_index: Tensor,
    kernel_size: int,
) -> Tensor:
    return grad_out.new_empty(kernel_size, x.size(1), grad_out.size(1))


@torch.library.register_fake('pyg::spline_weighting_backward_basis')
def spline_weighting_backward_basis(
    grad_out: Tensor,
    x: Tensor,
    weight: Tensor,
    weight_index: Tensor,
) -> Tensor:
    return grad_out.new_empty(grad_out.size(0), weight_index.size(1))


# Point clouds and clustering #################################################


@torch.library.register_fake('pyg::knn')
def knn(
    x: Tensor,
    y: Tensor,
    ptr_x: Optional[Tensor] = None,
    ptr_y: Optional[Tensor] = None,
    k: int = 1,
    cosine: bool = False,
    num_workers: int = 1,
) -> Tensor:
    return x.new_empty(2, new_dynamic_size(), dtype=torch.long)


@torch.library.register_fake('pyg::radius')
def radius(
    x: Tensor,
    y: Tensor,
    ptr_x: Optional[Tensor] = None,
    ptr_y: Optional[Tensor] = None,
    r: float = 1.0,
    max_num_neighbors: int = 32,
    num_workers: int = 1,
    ignore_same_index: bool = False,
) -> Tensor:
    return x.new_empty(2, new_dynamic_size(), dtype=torch.long)


@torch.library.register_fake('pyg::nearest')
def nearest(
    x: Tensor,
    y: Tensor,
    ptr_x: Optional[Tensor] = None,
    ptr_y: Optional[Tensor] = None,
) -> Tensor:
    return x.new_empty(x.size(0), dtype=torch.long)


@torch.library.register_fake('pyg::fps')
def fps(
    src: Tensor,
    ptr: Tensor,
    ratio: float = 0.5,
    random_start: bool = True,
) -> Tensor:
    return ptr.new_empty(new_dynamic_size())


@torch.library.register_fake('pyg::grid_cluster')
def grid_cluster(
    pos: Tensor,
    size: Tensor,
    start: Optional[Tensor] = None,
    end: Optional[Tensor] = None,
) -> Tensor:
    return pos.new_empty(pos.size(0), dtype=torch.long)


@torch.library.register_fake('pyg::graclus_cluster')
def graclus_cluster(
    rowptr: Tensor,
    col: Tensor,
    weight: Optional[Tensor] = None,
) -> Tensor:
    return rowptr.new_empty(rowptr.numel() - 1)


@torch.library.register_fake('pyg::metis')
def metis(
    rowptr: Tensor,
    col: Tensor,
    num_partitions: int,
    node_weight: Optional[Tensor] = None,
    edge_weight: Optional[Tensor] = None,
    recursive: bool = False,
) -> Tensor:
    return rowptr.new_empty(rowptr.numel() - 1)


@torch.library.register_fake('pyg::index_sort')
def index_sort(
    indices: Tensor,
    max: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    out = torch.empty_like(indices)
    return out, out.new_empty(out.shape, dtype=torch.long)


# Sampling ####################################################################


@torch.library.register_fake('pyg::edge_sample')
def edge_sample(
    start: Tensor,
    rowptr: Tensor,
    count: int = 0,
    factor: float = 1.0,
) -> Tensor:
    return start.new_empty(new_dynamic_size())


@torch.library.register_fake('pyg::random_walk')
def random_walk(
    rowptr: Tensor,
    col: Tensor,
    seed: Tensor,
    walk_length: int,
    p: float,
    q: float,
) -> Tensor:
    return rowptr.new_empty(seed.size(0), walk_length + 1)


@torch.library.register_fake('pyg::subgraph')
def subgraph(
    rowptr: Tensor,
    col: Tensor,
    nodes: Tensor,
    return_edge_id: bool,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    num_edges = new_dynamic_size()
    return (
        rowptr.new_empty(nodes.size(0) + 1),
        col.new_empty(num_edges),
        col.new_empty(num_edges) if return_edge_id else None,
    )
