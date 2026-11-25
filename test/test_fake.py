from typing import Any, List, Tuple

import pytest
import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    ShapeEnv,
    free_unbacked_symbols,
)

import pyg_lib  # noqa
from pyg_lib.testing import withCUDA

# Operators that cannot be traced:
UNSUPPORTED = {
    # Does not operate on tensors:
    'cuda_version',
    # Neighbor sampling returns (and its post-processing consumes) lists of
    # integers that depend on the data, which `int[]` cannot represent:
    'neighbor_sample',
    'hetero_neighbor_sample',
    'dist_neighbor_sample',
    'merge_sampler_outputs',
    'relabel_neighborhood',
    'hetero_relabel_neighborhood',
}


def get_samples() -> List[Any]:
    r"""Returns example inputs for every operator, some of them multiple times
    in order to cover arguments that change how the output is determined.
    """
    x = torch.randn(8, 3)
    index = torch.tensor([0, 0, 1, 1, 1, 3, 3, 3])
    index2d = index.view(-1, 1).expand(-1, 3)
    ptr = torch.tensor([0, 2, 5, 5, 8])
    batch_ptr = torch.tensor([0, 4, 8])
    rowptr = torch.tensor([0, 2, 4, 6, 8])  # A ring of 4 nodes.
    col = torch.tensor([1, 3, 0, 2, 1, 3, 0, 2])
    nodes = torch.tensor([0, 1])
    pseudo, kernel_size = torch.rand(8, 2), torch.tensor([5, 5])
    is_open_spline = torch.tensor([1, 1], dtype=torch.uint8)
    basis, weight_index = torch.rand(8, 4), torch.randint(25, (8, 4))
    weight, grad_out = torch.randn(25, 3, 6), torch.randn(8, 6)

    samples = {
        'scatter_sum': [(x, index2d, 0), (x, index2d, 0, None, 6)],
        'scatter_mul': [(x, index2d, 0), (x, index2d, 0, None, 6)],
        # The mean operators read the output size as an integer in order to
        # count elements, and thus require `dim_size` to be traceable:
        'scatter_mean': [(x, index2d, 0, None, 6)],
        'scatter_min': [(x, index2d, 0), (x, index2d, 0, None, 6)],
        'scatter_max': [(x, index2d, 0), (x, index2d, 0, None, 6)],
        'segment_sum_coo': [(x, index), (x, index, None, 6)],
        'segment_mean_coo': [(x, index, None, 6)],
        'segment_min_coo': [(x, index), (x, index, None, 6)],
        'segment_max_coo': [(x, index), (x, index, None, 6)],
        'segment_sum_csr': [(x, ptr)],
        'segment_mean_csr': [(x, ptr)],
        'segment_min_csr': [(x, ptr)],
        'segment_max_csr': [(x, ptr)],
        'gather_coo': [(x[:4], index)],
        'gather_csr': [(x[:4], ptr)],
        'softmax_csr': [(x, ptr)],
        'softmax_csr_backward': [(x, x, ptr)],
        'grouped_matmul': [([x, x.t()], [x.t(), x])],
        'segment_matmul': [(x, batch_ptr, torch.randn(2, 3, 6))],
        'sampled_op': [
            (x, x, index, index, 'add'),
            (x, x, None, index, 'sub'),
            (x, x, index, None, 'mul'),
            (x, x, None, None, 'div'),
            (weight_index, weight_index + 1, None, None, 'div'),
        ],
        'spline_basis': [(pseudo, kernel_size, is_open_spline, 1)],
        'spline_basis_backward': [
            (basis, pseudo, kernel_size, is_open_spline, 1),
        ],
        'spline_weighting': [(x, weight, basis, weight_index)],
        'spline_weighting_backward_x': [
            (grad_out, weight, basis, weight_index),
        ],
        'spline_weighting_backward_weight': [
            (grad_out, x, basis, weight_index, 25),
        ],
        'spline_weighting_backward_basis': [
            (grad_out, x, weight, weight_index),
        ],
        'knn': [(x, x, None, None, 2), (x, x, batch_ptr, batch_ptr, 2)],
        'radius': [(x, x), (x, x, batch_ptr, batch_ptr, 5.0)],
        'nearest': [(x, x[:4]), (x, x, batch_ptr, batch_ptr)],
        'fps': [(x, batch_ptr, 0.5, False), (x, batch_ptr, 0.25, False)],
        'grid_cluster': [(x, torch.ones(3))],
        'graclus_cluster': [(rowptr, col), (rowptr, col, torch.rand(8))],
        'index_sort': [(index,), (index, 4)],
        'edge_sample': [(nodes, rowptr), (nodes, rowptr, 2)],
        'random_walk': [(rowptr, col, nodes, 5, 1.0, 1.0)],
        'subgraph': [(rowptr, col, nodes, True), (rowptr, col, nodes, False)],
        'metis': [(rowptr, col, 2)],
    }

    return [
        pytest.param(name, args, id=f'{name}{i}')
        for name, list_of_args in samples.items()
        for i, args in enumerate(list_of_args)
    ]


def has_kernel(name: str, key: str) -> bool:
    return torch._C._dispatch_has_kernel_for_dispatch_key(f'pyg::{name}', key)


def to_device(name: str, args: Tuple[Any], device: torch.device) -> Tuple[Any]:
    if device.type == 'cuda' and not has_kernel(name, 'CUDA'):
        pytest.skip(f"'{name}' does not support CUDA")
    return pytree.tree_map_only(Tensor, lambda x: x.to(device), args)


def test_fake_is_complete() -> None:
    # Every operator either needs to be traceable or explicitly opt out. This
    # fails if an operator is added without a fake kernel and example inputs:
    operators = {
        schema.name.split('::')[1]
        for schema in torch._C._jit_get_all_schemas()
        if schema.name.startswith('pyg::')
    }
    supported = {param.values[0] for param in get_samples()}
    assert supported.isdisjoint(UNSUPPORTED)
    assert supported | UNSUPPORTED == operators


@withCUDA
@pytest.mark.parametrize('name,args', get_samples())
@pytest.mark.parametrize('static_shapes', [True, False])
def test_fake(
    name: str,
    args: Tuple[Any],
    static_shapes: bool,
    device: torch.device,
) -> None:
    if not static_shapes and has_kernel(name, 'Autograd'):
        pytest.skip(f"Autograd kernel of '{name}' requires static shapes")

    op = getattr(torch.ops.pyg, name)
    args = to_device(name, args, device)
    outs = pytree.tree_leaves(op(*args))

    shape_env = ShapeEnv(allow_dynamic_output_shape_ops=True)
    with FakeTensorMode(shape_env=shape_env) as mode:
        fake_args = pytree.tree_map_only(
            Tensor,
            lambda x: mode.from_tensor(x, static_shapes=static_shapes),
            args,
        )
        fake_outs = pytree.tree_leaves(op(*fake_args))

    # Tracing should not specialize on the input shapes:
    assert len(shape_env.guards) == 0

    assert len(fake_outs) == len(outs)
    for fake_out, out in zip(fake_outs, outs):
        if out is None:
            assert fake_out is None
            continue
        assert fake_out.dtype == out.dtype
        assert fake_out.device == out.device
        assert fake_out.dim() == out.dim()
        for fake_size, size in zip(fake_out.size(), out.size()):
            # Sizes are either known in advance or depend on the data:
            if len(free_unbacked_symbols(fake_size)) == 0:
                assert fake_size == size


class Module(torch.nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.op = getattr(torch.ops.pyg, name)

    def forward(self, *args: Any) -> Any:
        return self.op(*args)


@withCUDA
@pytest.mark.parametrize('name,args', get_samples())
@pytest.mark.parametrize('strict', [True, False])
def test_export(
    name: str,
    args: Tuple[Any],
    strict: bool,
    device: torch.device,
) -> None:
    module = Module(name)
    args = to_device(name, args, device)
    exported_module = torch.export.export(module, args, strict=strict).module()

    torch.manual_seed(12345)
    expected_outs = pytree.tree_leaves(module(*args))
    torch.manual_seed(12345)
    outs = pytree.tree_leaves(exported_module(*args))

    torch.testing.assert_close(outs, expected_outs)


def test_out() -> None:
    x = torch.randn(8, 3)
    index = torch.tensor([0, 0, 1, 1, 1, 3, 3, 3])
    args = (x, index.view(-1, 1).expand(-1, 3), 0, torch.zeros(4, 3))

    with pytest.raises(Exception, match="Cannot trace the 'out' argument"):
        torch.export.export(Module('scatter_sum'), args, strict=False)


# Backward passes that cannot be traced yet since their C++ implementation
# writes into `out`, reads a data-dependent output size or accesses the data:
NO_BACKWARD = {
    'segment_sum_coo0',
    'segment_mean_coo0',
    'segment_sum_csr0',
    'segment_mean_csr0',
    'gather_coo0',
    'gather_csr0',
    'segment_matmul0',
}


def get_differentiable_samples() -> List[Any]:
    samples = []
    for param in get_samples():
        name, args = param.values
        if not has_kernel(name, 'Autograd') or name.endswith('_backward'):
            continue
        tensors = [x for x in pytree.tree_leaves(args) if torch.is_tensor(x)]
        if not any(x.is_floating_point() for x in tensors):
            continue
        marks = [pytest.mark.xfail] if param.id in NO_BACKWARD else []
        samples.append(pytest.param(name, args, id=param.id, marks=marks))
    return samples


@pytest.mark.parametrize('name,args', get_differentiable_samples())
def test_backward(name: str, args: Tuple[Any]) -> None:
    # The backward pass is traced as well, and calls further operators:
    def get_grads(module: Any) -> List[Tensor]:
        leaves = pytree.tree_map_only(
            Tensor,
            lambda x: x.clone().requires_grad_(x.is_floating_point()),
            args,
        )
        outs = pytree.tree_leaves(module(*leaves))
        sum((2.0 * out).sum() for out in outs if out.requires_grad).backward()
        return [
            x.grad
            for x in pytree.tree_leaves(leaves)
            if isinstance(x, Tensor) and x.grad is not None
        ]

    torch._dynamo.reset()
    module = Module(name)
    compiled = torch.compile(module, fullgraph=True, backend='aot_eager')

    expected_grads = get_grads(module)
    assert len(expected_grads) > 0
    torch.testing.assert_close(get_grads(compiled), expected_grads)
