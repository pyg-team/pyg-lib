import pytest
import torch

import pyg_lib
from pyg_lib.testing import withCUDA


@withCUDA
@pytest.mark.parametrize('degree', [1, 2, 3])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_spline_conv_forward(
    degree: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    N, E, M_in, M_out, D = 4, 6, 8, 16, 2
    kernel_size_val = 5
    K = kernel_size_val**D

    x = torch.randn(N, M_in, dtype=dtype, device=device)
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 3], [1, 2, 0, 3, 0, 2]],
        dtype=torch.long,
        device=device,
    )
    pseudo = torch.rand(E, D, dtype=dtype, device=device)
    weight = torch.randn(K, M_in, M_out, dtype=dtype, device=device)
    kernel_size = torch.tensor(
        [kernel_size_val] * D,
        dtype=torch.long,
        device=device,
    )
    is_open_spline = torch.ones(D, dtype=torch.uint8, device=device)

    out = pyg_lib.ops.spline_conv(
        x,
        edge_index,
        pseudo,
        weight,
        kernel_size,
        is_open_spline,
        degree=degree,
    )
    assert out.shape == (N, M_out)
    assert out.dtype == dtype

    # Verify against manual computation:
    row, col = edge_index[0], edge_index[1]
    basis, wi = pyg_lib.ops.spline_basis(
        pseudo,
        kernel_size,
        is_open_spline,
        degree,
    )
    msg = pyg_lib.ops.spline_weighting(x[col], weight, basis, wi)
    ref = x.new_zeros(N, M_out)
    ref.scatter_add_(0, row.unsqueeze(1).expand_as(msg), msg)
    deg = x.new_zeros(N)
    deg.scatter_add_(0, row, x.new_ones(row.size(0)))
    deg = deg.clamp(min=1).unsqueeze(1)
    ref = ref / deg

    torch.testing.assert_close(out, ref)


@withCUDA
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_spline_conv_no_norm(
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    N, E, M_in, M_out, D = 4, 6, 8, 16, 2
    K = 25

    x = torch.randn(N, M_in, dtype=dtype, device=device)
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 3], [1, 2, 0, 3, 0, 2]],
        dtype=torch.long,
        device=device,
    )
    pseudo = torch.rand(E, D, dtype=dtype, device=device)
    weight = torch.randn(K, M_in, M_out, dtype=dtype, device=device)
    kernel_size = torch.tensor([5, 5], dtype=torch.long, device=device)
    is_open_spline = torch.ones(D, dtype=torch.uint8, device=device)

    out = pyg_lib.ops.spline_conv(
        x,
        edge_index,
        pseudo,
        weight,
        kernel_size,
        is_open_spline,
        norm=False,
    )
    assert out.shape == (N, M_out)

    # Without norm, verify no division by degree:
    row, col = edge_index[0], edge_index[1]
    basis, wi = pyg_lib.ops.spline_basis(
        pseudo,
        kernel_size,
        is_open_spline,
        1,
    )
    msg = pyg_lib.ops.spline_weighting(x[col], weight, basis, wi)
    ref = x.new_zeros(N, M_out)
    ref.scatter_add_(0, row.unsqueeze(1).expand_as(msg), msg)

    torch.testing.assert_close(out, ref)


@withCUDA
def test_spline_conv_root_weight_and_bias(device: torch.device) -> None:
    N, E, M_in, M_out, D = 4, 6, 8, 16, 2
    K = 25
    dtype = torch.float

    x = torch.randn(N, M_in, dtype=dtype, device=device)
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 3], [1, 2, 0, 3, 0, 2]],
        dtype=torch.long,
        device=device,
    )
    pseudo = torch.rand(E, D, dtype=dtype, device=device)
    weight = torch.randn(K, M_in, M_out, dtype=dtype, device=device)
    kernel_size = torch.tensor([5, 5], dtype=torch.long, device=device)
    is_open_spline = torch.ones(D, dtype=torch.uint8, device=device)
    root_weight = torch.randn(M_in, M_out, dtype=dtype, device=device)
    bias = torch.randn(M_out, dtype=dtype, device=device)

    out = pyg_lib.ops.spline_conv(
        x,
        edge_index,
        pseudo,
        weight,
        kernel_size,
        is_open_spline,
        root_weight=root_weight,
        bias=bias,
    )
    assert out.shape == (N, M_out)

    # Verify root_weight and bias are applied:
    out_no_extras = pyg_lib.ops.spline_conv(
        x,
        edge_index,
        pseudo,
        weight,
        kernel_size,
        is_open_spline,
    )
    expected = out_no_extras + x @ root_weight + bias
    torch.testing.assert_close(out, expected)


@withCUDA
def test_spline_conv_backward(device: torch.device) -> None:
    N, E, M_in, M_out, D = 4, 6, 4, 8, 2
    K = 25

    x = torch.randn(
        N,
        M_in,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 3], [1, 2, 0, 3, 0, 2]],
        dtype=torch.long,
        device=device,
    )
    pseudo = torch.rand(
        E,
        D,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    weight = torch.randn(
        K,
        M_in,
        M_out,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    kernel_size = torch.tensor([5, 5], dtype=torch.long, device=device)
    is_open_spline = torch.ones(D, dtype=torch.uint8, device=device)
    root_weight = torch.randn(
        M_in,
        M_out,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    bias = torch.randn(
        M_out,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )

    out = pyg_lib.ops.spline_conv(
        x,
        edge_index,
        pseudo,
        weight,
        kernel_size,
        is_open_spline,
        degree=1,
        root_weight=root_weight,
        bias=bias,
    )
    assert out.requires_grad
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert weight.grad is not None
    assert root_weight.grad is not None
    assert bias.grad is not None
