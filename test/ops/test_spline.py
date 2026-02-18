import math

import pytest
import torch

import pyg_lib

try:
    import torch_spline_conv
    HAS_TORCH_SPLINE_CONV = True
except ImportError:
    HAS_TORCH_SPLINE_CONV = False


def _basis_value(v: float, k_mod: int, degree: int) -> float:
    """Compute B-spline basis value for a single dimension."""
    if degree == 1:
        return 1.0 - v - k_mod + 2.0 * v * k_mod
    elif degree == 2:
        if k_mod == 0:
            return 0.5 * v * v - v + 0.5
        elif k_mod == 1:
            return -v * v + v + 0.5
        else:
            return 0.5 * v * v
    elif degree == 3:
        if k_mod == 0:
            return (1.0 - v)**3 / 6.0
        elif k_mod == 1:
            return (3.0 * v**3 - 6.0 * v**2 + 4.0) / 6.0
        elif k_mod == 2:
            return (-3.0 * v**3 + 3.0 * v**2 + 3.0 * v + 1.0) / 6.0
        else:
            return v**3 / 6.0


def _spline_basis_ref(
    pseudo: torch.Tensor,
    kernel_size: torch.Tensor,
    is_open_spline: torch.Tensor,
    degree: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    E, D = pseudo.shape
    S = (degree + 1)**D

    basis = torch.empty(E, S, dtype=pseudo.dtype)
    weight_index = torch.empty(E, S, dtype=torch.long)

    for e in range(E):
        for s in range(S):
            k = s
            wi = 0
            wi_offset = 1
            b = 1.0
            for d in range(D):
                k_mod = k % (degree + 1)
                k //= (degree + 1)

                v = pseudo[e, d].item()
                v *= (kernel_size[d].item() -
                      degree * is_open_spline[d].item())

                wi += ((int(v) + k_mod) % kernel_size[d].item() * wi_offset)
                wi_offset *= kernel_size[d].item()

                v -= math.floor(v)
                b *= _basis_value(v, k_mod, degree)

            basis[e, s] = b
            weight_index[e, s] = wi

    return basis, weight_index


def _spline_weigting_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    basis: torch.Tensor,
    weight_index: torch.Tensor,
) -> torch.Tensor:
    E = x.size(0)
    M_out = weight.size(2)
    S = basis.size(1)

    out = torch.zeros(E, M_out, dtype=x.dtype)
    for e in range(E):
        for s in range(S):
            b = basis[e, s]
            wi = weight_index[e, s]
            out[e] += b * (x[e].unsqueeze(-1) * weight[wi]).sum(dim=0)

    return out


@pytest.mark.parametrize('degree', [1, 2, 3])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_spline_basis_forward(degree: int, dtype: torch.dtype) -> None:
    E, D = 10, 3
    pseudo = torch.rand(E, D, dtype=dtype)
    kernel_size = torch.tensor([5, 5, 5], dtype=torch.long)
    is_open_spline = torch.tensor([1, 0, 1], dtype=torch.uint8)

    basis, wi = pyg_lib.ops.spline_basis(
        pseudo,
        kernel_size,
        is_open_spline,
        degree,
    )
    basis_ref, wi_ref = _spline_basis_ref(
        pseudo,
        kernel_size,
        is_open_spline,
        degree,
    )
    torch.testing.assert_close(basis, basis_ref)
    assert torch.equal(wi, wi_ref)


@pytest.mark.parametrize('degree', [1, 2, 3])
def test_spline_basis_backward(degree: int) -> None:
    E, D = 10, 3
    p = torch.rand(E, D, dtype=torch.double, requires_grad=True)
    kernel_size = torch.tensor([5, 5, 5], dtype=torch.long)
    is_open_spline = torch.tensor([1, 0, 1], dtype=torch.uint8)

    basis, wi = pyg_lib.ops.spline_basis(
        p,
        kernel_size,
        is_open_spline,
        degree,
    )
    assert basis.requires_grad
    assert not wi.requires_grad

    def fn(p_: torch.Tensor) -> torch.Tensor:
        return pyg_lib.ops.spline_basis(
            p_,
            kernel_size,
            is_open_spline,
            degree,
        )[0]

    torch.autograd.gradcheck(fn, p)


@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_spline_weighting_forward(dtype: torch.dtype) -> None:
    E, M_in, M_out = 10, 4, 8
    K = 25
    S = 4

    x = torch.randn(E, M_in, dtype=dtype)
    weight = torch.randn(K, M_in, M_out, dtype=dtype)
    basis = torch.rand(E, S, dtype=dtype)
    weight_index = torch.randint(0, K, (E, S), dtype=torch.long)

    out = pyg_lib.ops.spline_weighting(x, weight, basis, weight_index)
    out_ref = _spline_weigting_ref(x, weight, basis, weight_index)
    torch.testing.assert_close(out, out_ref)


def test_spline_weighting_backward() -> None:
    E, M_in, M_out = 10, 4, 8
    K = 25
    S = 4

    x = torch.randn(E, M_in, dtype=torch.double, requires_grad=True)
    weight = torch.randn(K, M_in, M_out, dtype=torch.double,
                         requires_grad=True)
    basis = torch.rand(E, S, dtype=torch.double, requires_grad=True)
    weight_index = torch.randint(0, K, (E, S), dtype=torch.long)

    out = pyg_lib.ops.spline_weighting(x, weight, basis, weight_index)
    assert out.requires_grad

    def fn(
        x_: torch.Tensor,
        w_: torch.Tensor,
        b_: torch.Tensor,
    ) -> torch.Tensor:
        return pyg_lib.ops.spline_weighting(x_, w_, b_, weight_index)

    torch.autograd.gradcheck(fn, (x, weight, basis))


@pytest.mark.skipif(
    not HAS_TORCH_SPLINE_CONV,
    reason='torch_spline_conv not available',
)
@pytest.mark.parametrize('degree', [1, 2, 3])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_parity_spline_basis(
    degree: int,
    dtype: torch.dtype,
) -> None:
    E, D = 10, 3
    pseudo = torch.rand(E, D, dtype=dtype)
    kernel_size = torch.tensor([5, 5, 5], dtype=torch.long)
    is_open_spline = torch.tensor([1, 0, 1], dtype=torch.uint8)

    basis, wi = _spline_basis_ref(pseudo, kernel_size, is_open_spline, degree)
    basis_ref, wi_ref = torch_spline_conv.spline_basis(
        pseudo,
        kernel_size,
        is_open_spline,
        degree,
    )
    torch.testing.assert_close(basis, basis_ref)
    assert torch.equal(wi, wi_ref)


@pytest.mark.skipif(
    not HAS_TORCH_SPLINE_CONV,
    reason='torch_spline_conv not available',
)
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_parity_spline_weighting(dtype: torch.dtype) -> None:
    E, M_in, M_out = 10, 4, 8
    K = 25
    S = 4

    x = torch.randn(E, M_in, dtype=dtype)
    weight = torch.randn(K, M_in, M_out, dtype=dtype)
    basis = torch.rand(E, S, dtype=dtype)
    weight_index = torch.randint(0, K, (E, S), dtype=torch.long)

    out = _spline_weigting_ref(x, weight, basis, weight_index)
    out_ref = torch_spline_conv.spline_weighting(
        x,
        weight,
        basis,
        weight_index,
    )
    torch.testing.assert_close(out, out_ref)
