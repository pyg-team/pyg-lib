r"""Fake kernels for all :obj:`pyg::*` operators.

A fake kernel describes the outputs of an operator (shape, data type and
device) without computing them. :pytorch:`PyTorch` requires it in order to
trace an operator with data-less tensors, which is how :obj:`torch.export` and
:obj:`torch.compile` capture a model.
"""

import functools
import inspect
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor


def register_fake(*names: str) -> Callable:
    r"""Registers a function as the fake kernel of the given operators.

    The function is called with the arguments of the operator **by name**,
    with defaults taken from the operator schema. As such, it only needs to
    declare the arguments that determine the outputs.
    """

    def decorator(func: Callable) -> Callable:
        params = list(inspect.signature(func).parameters)

        for name in names:
            # The operator is missing if the shared library could not be
            # loaded, e.g., when building the documentation:
            op = getattr(torch.ops.pyg, name, None)
            if op is None:
                continue

            def fake(*args, _schema=op.default._schema):
                kwargs = {a.name: a.default_value for a in _schema.arguments}
                kwargs.update(zip(kwargs, args))
                return func(*[kwargs[param] for param in params])

            torch.library.register_fake(f'pyg::{name}')(fake)

        return func

    return decorator


def dynamic_size() -> torch.SymInt:
    r"""Returns a size that depends on the data, i.e. is unknown in advance."""
    return torch.library.get_ctx().new_dynamic_size()


def resize(src: Tensor, dim: int, size: int) -> Tensor:
    r"""Returns a tensor like :obj:`src` of the given size in :obj:`dim`."""
    sizes = list(src.shape)
    sizes[dim] = size
    return src.new_empty(sizes)


def with_arg(func: Callable) -> Callable:
    r"""Additionally returns the index of each output element."""

    @functools.wraps(func)  # Keeps the argument names to bind by.
    def wrapper(*args) -> Tuple[Tensor, Tensor]:
        out = func(*args)
        return out, out.new_empty(out.shape, dtype=torch.long)

    return wrapper


# Reductions ##################################################################


def check_out(out: Optional[Tensor]) -> None:
    # Writing into `out` cannot be traced since the operator schemas do not
    # declare that it is modified in-place:
    if out is not None:
        raise NotImplementedError("Cannot trace the 'out' argument")


def scatter(src, index, dim, out, dim_size) -> Tensor:
    check_out(out)
    # Without `dim_size`, the output size is given by `index.max() + 1`:
    size = dynamic_size() if dim_size is None else dim_size
    return resize(src, dim, size)


register_fake('scatter_sum', 'scatter_mul', 'scatter_mean')(scatter)
register_fake('scatter_min', 'scatter_max')(with_arg(scatter))


def segment_coo(src, index, out, dim_size) -> Tensor:
    return scatter(src, index, index.dim() - 1, out, dim_size)


register_fake('segment_sum_coo', 'segment_mean_coo')(segment_coo)
register_fake('segment_min_coo', 'segment_max_coo')(with_arg(segment_coo))


def segment_csr(src, indptr, out) -> Tensor:
    check_out(out)
    size = torch.sym_max(indptr.size(-1) - 1, 0)
    return resize(src, indptr.dim() - 1, size)


register_fake('segment_sum_csr', 'segment_mean_csr')(segment_csr)
register_fake('segment_min_csr', 'segment_max_csr')(with_arg(segment_csr))


@register_fake('gather_coo')
def gather_coo(src, index, out) -> Tensor:
    check_out(out)
    return resize(src, index.dim() - 1, index.size(-1))


@register_fake('gather_csr')
def gather_csr(src, indptr, out) -> Tensor:
    check_out(out)
    # The output size is given by the last element in `indptr`:
    return resize(src, indptr.dim() - 1, dynamic_size())


@register_fake('softmax_csr')
def softmax_csr(src) -> Tensor:
    return torch.empty_like(src)


@register_fake('softmax_csr_backward')
def softmax_csr_backward(out) -> Tensor:
    return torch.empty_like(out)


# Matrix multiplication #######################################################


@register_fake('grouped_matmul')
def grouped_matmul(input: List[Tensor], other: List[Tensor]) -> List[Tensor]:
    return [x.new_empty(x.size(0), y.size(-1)) for x, y in zip(input, other)]


@register_fake('segment_matmul')
def segment_matmul(input, other) -> Tensor:
    return input.new_empty(input.size(0), other.size(-1))


@register_fake('sampled_op')
def sampled_op(left, right, left_index, right_index, op) -> Tensor:
    if left_index is not None:
        left = left.index_select(0, left_index)
    if right_index is not None:
        right = right.index_select(0, right_index)
    out = getattr(torch, op)(left, right)
    # The CUDA kernel does not perform type promotion:
    return out.to(left.dtype) if out.is_cuda else out


# Spline ######################################################################


@register_fake('spline_basis')
def spline_basis(pseudo, kernel_size, degree) -> Tuple[Tensor, Tensor]:
    size = (pseudo.size(0), (degree + 1) ** pseudo.size(1))
    return pseudo.new_empty(size), kernel_size.new_empty(size)


@register_fake('spline_basis_backward')
def spline_basis_backward(pseudo) -> Tensor:
    return torch.empty_like(pseudo)


@register_fake('spline_weighting')
def spline_weighting(x, weight) -> Tensor:
    return x.new_empty(x.size(0), weight.size(2))


@register_fake('spline_weighting_backward_x')
def spline_weighting_backward_x(grad_out, weight) -> Tensor:
    return grad_out.new_empty(grad_out.size(0), weight.size(1))


@register_fake('spline_weighting_backward_weight')
def spline_weighting_backward_weight(grad_out, x, kernel_size) -> Tensor:
    return grad_out.new_empty(kernel_size, x.size(1), grad_out.size(1))


@register_fake('spline_weighting_backward_basis')
def spline_weighting_backward_basis(grad_out, weight_index) -> Tensor:
    return grad_out.new_empty(grad_out.size(0), weight_index.size(1))


# Point clouds and clustering #################################################


@register_fake('knn', 'radius')
def knn(x) -> Tensor:
    return x.new_empty(2, dynamic_size(), dtype=torch.long)


@register_fake('nearest')
def nearest(x) -> Tensor:
    return x.new_empty(x.size(0), dtype=torch.long)


@register_fake('fps')
def fps(ptr) -> Tensor:
    return ptr.new_empty(dynamic_size())


@register_fake('grid_cluster')
def grid_cluster(pos) -> Tensor:
    return pos.new_empty(pos.size(0), dtype=torch.long)


@register_fake('graclus_cluster', 'metis')
def graclus_cluster(rowptr) -> Tensor:
    return rowptr.new_empty(rowptr.numel() - 1)


@register_fake('index_sort')
@with_arg
def index_sort(indices) -> Tensor:
    return torch.empty_like(indices)


# Sampling ####################################################################


@register_fake('edge_sample')
def edge_sample(start) -> Tensor:
    return start.new_empty(dynamic_size())


@register_fake('random_walk')
def random_walk(rowptr, seed, walk_length) -> Tensor:
    return rowptr.new_empty(seed.size(0), walk_length + 1)


@register_fake('subgraph')
def subgraph(
    rowptr,
    col,
    nodes,
    return_edge_id,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    num_edges = dynamic_size()
    return (
        rowptr.new_empty(nodes.size(0) + 1),
        col.new_empty(num_edges),
        col.new_empty(num_edges) if return_edge_id else None,
    )
