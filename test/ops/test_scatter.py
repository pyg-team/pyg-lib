import pytest
import torch

import pyg_lib
from pyg_lib.testing import withCUDA


def _broadcast_index(
    index: torch.Tensor,
    src: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Broadcasts a 1-D :obj:`index` to the shape of :obj:`src` along
    :obj:`dim`. Mirrors the helper used by upstream pytorch_scatter and the
    C++ ``broadcast`` util in ``pyg_lib/csrc/ops/utils.h``.
    """
    if index.dim() == src.dim():
        return index
    if dim < 0:
        dim = dim + src.dim()
    size = [1] * src.dim()
    size[dim] = -1
    return index.view(size).expand_as(src)


def _scatter_sum_ref(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor = None,
    dim_size: int = None,
) -> torch.Tensor:
    """Pure-PyTorch reference implementation of :func:`scatter_sum`.

    Mirrors the contract of ``pyg_lib.ops.scatter_sum``:
      * Index is broadcast to ``src`` along ``dim``.
      * If ``out`` is :obj:`None`, an output of the appropriate shape is
        zero-initialized.
      * If ``out`` is provided, the op accumulates into it (no zeroing).
      * If ``dim_size`` is :obj:`None`, infer from ``index.max() + 1``.
    """
    if dim < 0:
        dim = dim + src.dim()
    bcast_index = _broadcast_index(index, src, dim)
    if out is None:
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        size = list(src.size())
        size[dim] = dim_size
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
    out.scatter_add_(dim, bcast_index, src)
    return out


def test_scatter_add_is_scatter_sum_alias():
    assert pyg_lib.ops.scatter_add is pyg_lib.ops.scatter_sum


@withCUDA
@pytest.mark.parametrize(
    'dtype',
    [
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    ],
)
def test_scatter_sum_forward_dtypes(dtype, device):
    if device.type == 'cpu' and dtype == torch.float16:
        # CPU half-precision arithmetic is supported but tolerances are loose;
        # we still run it to check parity.
        pass
    if dtype in (torch.int32, torch.int64):
        src = torch.randint(-10, 10, (8, 4), dtype=dtype, device=device)
    else:
        src = torch.randn(8, 4, dtype=dtype, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0], device=device)

    out = pyg_lib.ops.scatter_sum(src, index, dim=0)
    expected = _scatter_sum_ref(src, index, dim=0)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_sum_forward_dim_neg1(device):
    src = torch.randn(3, 8, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0], device=device)

    out = pyg_lib.ops.scatter_sum(src, index, dim=-1)
    expected = _scatter_sum_ref(src, index, dim=-1)
    assert out.size() == (3, 4)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_sum_forward_dim_nonneg(device):
    src = torch.randn(8, 4, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0], device=device)

    out = pyg_lib.ops.scatter_sum(src, index, dim=0)
    expected = _scatter_sum_ref(src, index, dim=0)
    assert out.size() == (4, 4)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_sum_forward_dim_middle(device):
    src = torch.randn(3, 6, 5, device=device)
    index = torch.tensor([0, 2, 1, 0, 2, 1], device=device)

    out = pyg_lib.ops.scatter_sum(src, index, dim=1)
    expected = _scatter_sum_ref(src, index, dim=1)
    assert out.size() == (3, 3, 5)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_sum_broadcasting_1d_index(device):
    """1-D ``index`` broadcasts to 2-D ``src`` along ``dim``."""
    src = torch.randn(6, 4, device=device)
    index = torch.tensor([0, 1, 0, 2, 1, 2], device=device)

    out = pyg_lib.ops.scatter_sum(src, index, dim=0)
    expected = _scatter_sum_ref(src, index, dim=0)
    assert out.size() == (3, 4)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_sum_dim_size_auto_infer(device):
    src = torch.randn(6, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    out = pyg_lib.ops.scatter_sum(src, index, dim=-1)
    # Auto-inferred dim_size = index.max() + 1 = 4
    assert out.size(-1) == 4
    expected = _scatter_sum_ref(src, index, dim=-1)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_sum_dim_size_explicit(device):
    src = torch.randn(6, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    # Larger than implicit dim_size — trailing buckets are empty (zeros).
    out = pyg_lib.ops.scatter_sum(src, index, dim=-1, dim_size=6)
    assert out.size(-1) == 6
    expected = _scatter_sum_ref(src, index, dim=-1, dim_size=6)
    torch.testing.assert_close(out, expected)
    # Empty buckets should be exactly zero.
    torch.testing.assert_close(out[4:], torch.zeros(2, device=device))


@withCUDA
def test_scatter_sum_out_accumulates(device):
    """When ``out`` is provided, the op accumulates into it without zeroing."""
    src = torch.randn(6, 4, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    out_init = torch.randn(4, 4, device=device)
    out = out_init.clone()
    result = pyg_lib.ops.scatter_sum(src, index, dim=0, out=out)

    # Accumulate semantics: result == out_init + scatter_sum into zeros.
    delta = _scatter_sum_ref(src, index, dim=0, dim_size=4)  # zero-initialized
    expected = out_init + delta
    torch.testing.assert_close(result, expected)
    # The op should also have updated ``out`` in-place.
    torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_sum_empty_input(device):
    """An empty source with an empty index should yield an empty output."""
    src = torch.empty(0, 4, device=device)
    index = torch.empty(0, dtype=torch.long, device=device)

    out = pyg_lib.ops.scatter_sum(src, index, dim=0, dim_size=3)
    assert out.size() == (3, 4)
    torch.testing.assert_close(out, torch.zeros(3, 4, device=device))


@withCUDA
def test_scatter_sum_backward_gradcheck(device):
    src = torch.randn(
        6,
        3,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    def fn(s):
        return pyg_lib.ops.scatter_sum(s, index, dim=0)

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_scatter_sum_backward_gradcheck_dim_middle(device):
    src = torch.randn(
        2,
        6,
        3,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    index = torch.tensor([0, 1, 0, 1, 2, 1], device=device)

    def fn(s):
        return pyg_lib.ops.scatter_sum(s, index, dim=1)

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_scatter_sum_backward_gradcheck_with_dim_size(device):
    """Gradcheck with an explicit (larger) ``dim_size`` exercising empty
    buckets in the output.
    """
    src = torch.randn(6, dtype=torch.double, device=device, requires_grad=True)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    def fn(s):
        return pyg_lib.ops.scatter_sum(s, index, dim=-1, dim_size=6)

    assert torch.autograd.gradcheck(fn, (src,))


# ---------------------------------------------------------------------------
# scatter_mul
# ---------------------------------------------------------------------------


def _scatter_mul_ref(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor = None,
    dim_size: int = None,
) -> torch.Tensor:
    """Pure-PyTorch reference implementation of :func:`scatter_mul`.

    Mirrors the contract of ``pyg_lib.ops.scatter_mul``:
      * Index is broadcast to ``src`` along ``dim``.
      * If ``out`` is :obj:`None`, an output of the appropriate shape is
        initialized to **ones** (not zeros).
      * If ``out`` is provided, the op multiplies into it (no ones-init).
      * If ``dim_size`` is :obj:`None`, infer from ``index.max() + 1``.

    Implemented via an explicit per-element scalar walk along ``dim`` so we
    avoid relying on ``Tensor.scatter_reduce_(reduce="prod")`` whose
    initialization semantics differ from upstream's ones-init.
    """
    if dim < 0:
        dim = dim + src.dim()
    bcast_index = _broadcast_index(index, src, dim)
    if out is None:
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        size = list(src.size())
        size[dim] = dim_size
        out = torch.ones(size, dtype=src.dtype, device=src.device)
    else:
        out = out.clone()  # don't mutate caller's tensor in the reference

    # Walk over ``src`` along ``dim`` and multiply each slice into ``out``.
    other_shape = list(src.size())
    del other_shape[dim]
    for i in range(src.size(dim)):
        src_slice = src.select(dim, i)
        idx_slice = bcast_index.select(dim, i)
        flat_src = src_slice.reshape(-1)
        flat_idx = idx_slice.reshape(-1)
        for k in range(flat_idx.numel()):
            j = int(flat_idx[k].item())
            # Recover the multi-index for the "other" dims.
            coord = []
            rem = k
            for s in reversed(other_shape):
                coord.append(rem % s)
                rem //= s
            coord = list(reversed(coord))
            out_idx = list(coord)
            out_idx.insert(dim, j)
            out[tuple(out_idx)] = out[tuple(out_idx)] * flat_src[k]
    return out


@withCUDA
@pytest.mark.parametrize(
    'dtype',
    [
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    ],
)
def test_scatter_mul_forward_dtypes(dtype, device):
    if dtype in (torch.int32, torch.int64):
        # Keep values small so the product doesn't blow up integer ranges.
        src = torch.randint(-3, 4, (8, 4), dtype=dtype, device=device)
    else:
        # Keep values O(1) so fp16/bf16 products stay well within range.
        src = torch.randn(8, 4, dtype=dtype, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0], device=device)

    out = pyg_lib.ops.scatter_mul(src, index, dim=0)
    expected = _scatter_mul_ref(src, index, dim=0)
    # Looser tolerances for low-precision floats; chained multiplies amplify
    # rounding error relative to a single scatter_add pass.
    if dtype in (torch.float16, torch.bfloat16):
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
    else:
        torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_mul_forward_dim_neg1(device):
    src = torch.randn(3, 8, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0], device=device)

    out = pyg_lib.ops.scatter_mul(src, index, dim=-1)
    expected = _scatter_mul_ref(src, index, dim=-1)
    assert out.size() == (3, 4)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_mul_forward_dim_nonneg(device):
    src = torch.randn(8, 4, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0], device=device)

    out = pyg_lib.ops.scatter_mul(src, index, dim=0)
    expected = _scatter_mul_ref(src, index, dim=0)
    assert out.size() == (4, 4)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_mul_forward_dim_middle(device):
    src = torch.randn(3, 6, 5, device=device)
    index = torch.tensor([0, 2, 1, 0, 2, 1], device=device)

    out = pyg_lib.ops.scatter_mul(src, index, dim=1)
    expected = _scatter_mul_ref(src, index, dim=1)
    assert out.size() == (3, 3, 5)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_mul_broadcasting_1d_index(device):
    """1-D ``index`` broadcasts to 2-D ``src`` along ``dim``."""
    src = torch.randn(6, 4, device=device)
    index = torch.tensor([0, 1, 0, 2, 1, 2], device=device)

    out = pyg_lib.ops.scatter_mul(src, index, dim=0)
    expected = _scatter_mul_ref(src, index, dim=0)
    assert out.size() == (3, 4)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_mul_dim_size_auto_infer(device):
    src = torch.randn(6, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    out = pyg_lib.ops.scatter_mul(src, index, dim=-1)
    # Auto-inferred dim_size = index.max() + 1 = 4
    assert out.size(-1) == 4
    expected = _scatter_mul_ref(src, index, dim=-1)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_mul_dim_size_explicit(device):
    """Explicit ``dim_size`` larger than implicit — trailing buckets remain at
    the ones-init (no src elements multiply into them).
    """
    src = torch.randn(6, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    out = pyg_lib.ops.scatter_mul(src, index, dim=-1, dim_size=6)
    assert out.size(-1) == 6
    expected = _scatter_mul_ref(src, index, dim=-1, dim_size=6)
    torch.testing.assert_close(out, expected)
    # Empty buckets keep the ones-init (NOT zero, unlike scatter_sum).
    torch.testing.assert_close(out[4:], torch.ones(2, device=device))


@withCUDA
def test_scatter_mul_out_multiplies_into_buffer(device):
    """When ``out`` is provided, mul into it without ones-init."""
    src = torch.randn(6, 4, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    out_init = torch.randn(4, 4, device=device)
    out = out_init.clone()
    result = pyg_lib.ops.scatter_mul(src, index, dim=0, out=out)

    # Multiply-into semantics: result == out_init * scatter_mul into ones.
    factor = _scatter_mul_ref(
        src,
        index,
        dim=0,
        dim_size=4,
    )  # ones-initialized
    expected = out_init * factor
    torch.testing.assert_close(result, expected)
    # The op should also have updated ``out`` in-place.
    torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_mul_empty_input(device):
    """An empty source with an empty index should yield ones-initialised
    output (no multiplies happened).
    """
    src = torch.empty(0, 4, device=device)
    index = torch.empty(0, dtype=torch.long, device=device)

    out = pyg_lib.ops.scatter_mul(src, index, dim=0, dim_size=3)
    assert out.size() == (3, 4)
    torch.testing.assert_close(out, torch.ones(3, 4, device=device))


@withCUDA
def test_scatter_mul_backward_gradcheck(device):
    # Avoid src == 0 in this base gradcheck — the upstream-matched
    # ``where(src != 0, ..., 0)`` rule deliberately deviates from the true
    # mathematical gradient at zeros, which would otherwise trip finite-diff.
    src = torch.randn(
        6,
        3,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    # Push values away from zero so the analytical == numerical comparison
    # uses the smooth branch of the autograd rule everywhere.
    with torch.no_grad():
        src.add_(src.sign() * 0.5)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    def fn(s):
        return pyg_lib.ops.scatter_mul(s, index, dim=0)

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_scatter_mul_backward_gradcheck_dim_middle(device):
    src = torch.randn(
        2,
        6,
        3,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    with torch.no_grad():
        src.add_(src.sign() * 0.5)
    index = torch.tensor([0, 1, 0, 1, 2, 1], device=device)

    def fn(s):
        return pyg_lib.ops.scatter_mul(s, index, dim=1)

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_scatter_mul_backward_gradcheck_with_dim_size(device):
    """Gradcheck with an explicit (larger) ``dim_size`` exercising empty
    buckets in the output. Empty buckets stay at the ones-init; their
    gradient w.r.t. ``src`` is zero because no ``src`` element flows into them.
    """
    src = torch.randn(6, dtype=torch.double, device=device, requires_grad=True)
    with torch.no_grad():
        src.add_(src.sign() * 0.5)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    def fn(s):
        return pyg_lib.ops.scatter_mul(s, index, dim=-1, dim_size=6)

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_scatter_mul_gradient_at_src_zero(device):
    """The mathematical gradient of a product is unstable at zero entries;
    upstream returns zero (we replicate via ``torch.where(src != 0, ...)``).

    Verify that the returned gradient at positions where ``src == 0`` is
    finite (zero), not NaN/inf, and that the gradient at non-zero positions
    is unaffected.
    """
    # Mix zeros into the source.
    src = torch.tensor(
        [2.0, 0.0, 3.0, 0.0, 5.0, 0.0],
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    index = torch.tensor([0, 0, 1, 1, 2, 2], device=device)

    out = pyg_lib.ops.scatter_mul(src, index, dim=0)
    # Sum so we have a scalar loss; ``grad_out`` is all-ones.
    out.sum().backward()

    grad = src.grad
    assert grad is not None
    # No NaNs and no infinities anywhere.
    assert torch.isfinite(grad).all(), (
        f'grad contains non-finite values: {grad}'
    )
    # Positions where src == 0 must have grad == 0 (the upstream convention).
    zero_mask = src.detach() == 0
    torch.testing.assert_close(
        grad[zero_mask],
        torch.zeros_like(grad[zero_mask]),
    )
    # Sanity-check non-zero positions: for bucket {2, 0} with grad_out=1,
    # grad_src for src==2 = (grad_out * out_bucket) / src_at_2 where
    # out_bucket = 2 * 0 = 0, so grad = 0. Same for all buckets containing
    # a zero entry. So actually all grads should be zero here.
    # Construct a different fixture where the non-zero grad path matters:
    src2 = torch.tensor(
        [2.0, 0.0, 3.0, 4.0, 5.0, 6.0],
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    index2 = torch.tensor([0, 0, 1, 1, 2, 2], device=device)
    out2 = pyg_lib.ops.scatter_mul(src2, index2, dim=0)
    out2.sum().backward()
    grad2 = src2.grad
    assert grad2 is not None
    assert torch.isfinite(grad2).all()
    # src2[1] == 0 -> grad2[1] must be 0 by the upstream rule.
    torch.testing.assert_close(
        grad2[1],
        torch.zeros((), dtype=torch.double, device=device),
    )
    # Bucket {3, 4}: product 12; grad[2] = 12/3 = 4, grad[3] = 12/4 = 3.
    torch.testing.assert_close(
        grad2[2],
        torch.tensor(4.0, dtype=torch.double, device=device),
    )
    torch.testing.assert_close(
        grad2[3],
        torch.tensor(3.0, dtype=torch.double, device=device),
    )
    # Bucket {5, 6}: product 30; grad[4] = 30/5 = 6, grad[5] = 30/6 = 5.
    torch.testing.assert_close(
        grad2[4],
        torch.tensor(6.0, dtype=torch.double, device=device),
    )
    torch.testing.assert_close(
        grad2[5],
        torch.tensor(5.0, dtype=torch.double, device=device),
    )
