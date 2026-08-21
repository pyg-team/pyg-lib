import pytest
import torch

import pyg_lib
from pyg_lib.testing import withCUDA, withSeed


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
    # Seed for deterministic random inputs across dtypes.
    torch.manual_seed(0)
    if dtype in (torch.int32, torch.int64):
        src = torch.randint(-10, 10, (8, 4), dtype=dtype, device=device)
    else:
        src = torch.randn(8, 4, dtype=dtype, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0], device=device)

    out = pyg_lib.ops.scatter_sum(src, index, dim=0)
    expected = _scatter_sum_ref(src, index, dim=0)
    # Half/bfloat16 accumulation drifts further than the default rtol allows.
    if dtype in (torch.float16, torch.bfloat16):
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
    else:
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


# ---------------------------------------------------------------------------
# scatter_mean
# ---------------------------------------------------------------------------


def _scatter_mean_ref(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor = None,
    dim_size: int = None,
) -> torch.Tensor:
    """Pure-PyTorch reference implementation of :func:`scatter_mean`.

    Mirrors the contract of ``pyg_lib.ops.scatter_mean``:
      * Index is broadcast to ``src`` along ``dim``.
      * Computes scatter_sum of ``src`` and divides by per-bucket counts.
      * Empty buckets (count == 0) are masked-filled to 1 to avoid div-by-0;
        the numerator is also zero there, so the result is zero.
      * Floats use true-division; integers use floor-division (matches the
        upstream ``div(_, rounding_mode="floor")`` semantics).
      * If ``out`` is :obj:`None`, an output of the appropriate shape is
        zero-initialised before the scatter_sum.
      * If ``out`` is provided, the sum accumulates into it *before* division.
      * If ``dim_size`` is :obj:`None`, infer from ``index.max() + 1``.
    """
    if dim < 0:
        dim = dim + src.dim()
    bcast_index = _broadcast_index(index, src, dim)

    # Numerator: scatter_sum into (a clone of) ``out``.
    if out is None:
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        size = list(src.size())
        size[dim] = dim_size
        numer = torch.zeros(size, dtype=src.dtype, device=src.device)
    else:
        # Don't mutate the caller's tensor in the reference.
        numer = out.clone()
    numer.scatter_add_(dim, bcast_index, src)

    # Denominator: per-bucket counts. Use a 1-D scatter_sum-of-ones along
    # ``dim``; then broadcast to ``numer.shape`` for the division.
    ones = torch.ones(src.size(dim), dtype=src.dtype, device=src.device)
    count = torch.zeros(numer.size(dim), dtype=src.dtype, device=src.device)
    count.scatter_add_(0, index, ones)
    count.masked_fill_(count < 1, 1)
    # Broadcast count along ``dim`` to ``numer.shape``.
    view = [1] * numer.dim()
    view[dim] = -1
    count = count.view(view).expand_as(numer)

    if numer.is_floating_point():
        return numer / count
    else:
        return torch.div(numer, count, rounding_mode='floor')


@withCUDA
@pytest.mark.parametrize(
    'dtype',
    [
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    ],
)
def test_scatter_mean_forward_dtypes(dtype, device):
    src = torch.randn(8, 4, dtype=dtype, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0], device=device)

    out = pyg_lib.ops.scatter_mean(src, index, dim=0)
    expected = _scatter_mean_ref(src, index, dim=0)
    if dtype in (torch.float16, torch.bfloat16):
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
    else:
        torch.testing.assert_close(out, expected)


@withCUDA
@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
def test_scatter_mean_forward_integer_floor_div(dtype, device):
    """Integer dtype: per-bucket result is the floor of sum/count."""
    src = torch.randint(-10, 10, (8, 4), dtype=dtype, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0], device=device)

    out = pyg_lib.ops.scatter_mean(src, index, dim=0)
    expected = _scatter_mean_ref(src, index, dim=0)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_mean_forward_dim_neg1(device):
    src = torch.randn(3, 8, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0], device=device)

    out = pyg_lib.ops.scatter_mean(src, index, dim=-1)
    expected = _scatter_mean_ref(src, index, dim=-1)
    assert out.size() == (3, 4)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_mean_forward_dim_nonneg(device):
    src = torch.randn(8, 4, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0], device=device)

    out = pyg_lib.ops.scatter_mean(src, index, dim=0)
    expected = _scatter_mean_ref(src, index, dim=0)
    assert out.size() == (4, 4)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_mean_forward_dim_middle(device):
    src = torch.randn(3, 6, 5, device=device)
    index = torch.tensor([0, 2, 1, 0, 2, 1], device=device)

    out = pyg_lib.ops.scatter_mean(src, index, dim=1)
    expected = _scatter_mean_ref(src, index, dim=1)
    assert out.size() == (3, 3, 5)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_mean_broadcasting_1d_index(device):
    """1-D ``index`` broadcasts to 2-D ``src`` along ``dim``."""
    src = torch.randn(6, 4, device=device)
    index = torch.tensor([0, 1, 0, 2, 1, 2], device=device)

    out = pyg_lib.ops.scatter_mean(src, index, dim=0)
    expected = _scatter_mean_ref(src, index, dim=0)
    assert out.size() == (3, 4)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_mean_dim_size_auto_infer(device):
    src = torch.randn(6, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    out = pyg_lib.ops.scatter_mean(src, index, dim=-1)
    # Auto-inferred dim_size = index.max() + 1 = 4
    assert out.size(-1) == 4
    expected = _scatter_mean_ref(src, index, dim=-1)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_mean_dim_size_explicit_empty_buckets(device):
    """Explicit ``dim_size`` larger than implicit — trailing buckets are empty
    and must yield exactly zero (count masked-filled to 1, then 0/1 = 0).
    This is the key contract that distinguishes scatter_mean from a naive
    sum/count which would produce NaN.
    """
    src = torch.randn(6, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    out = pyg_lib.ops.scatter_mean(src, index, dim=-1, dim_size=6)
    assert out.size(-1) == 6
    expected = _scatter_mean_ref(src, index, dim=-1, dim_size=6)
    torch.testing.assert_close(out, expected)
    # Empty trailing buckets at positions 4, 5 must be exactly zero, not NaN.
    torch.testing.assert_close(out[4:], torch.zeros(2, device=device))
    assert torch.isfinite(out).all(), (
        f'scatter_mean produced non-finite values: {out}'
    )


@withCUDA
def test_scatter_mean_empty_bucket_in_middle(device):
    """Empty buckets anywhere (not just trailing) must yield zero, not NaN."""
    # index skips bucket 2 entirely.
    src = torch.randn(6, 3, device=device)
    index = torch.tensor([0, 1, 0, 1, 3, 3], device=device)

    out = pyg_lib.ops.scatter_mean(src, index, dim=0, dim_size=4)
    assert out.size() == (4, 3)
    expected = _scatter_mean_ref(src, index, dim=0, dim_size=4)
    torch.testing.assert_close(out, expected)
    # Bucket 2 has no contributors — expect a zero row.
    torch.testing.assert_close(out[2], torch.zeros(3, device=device))
    assert torch.isfinite(out).all()


@withCUDA
def test_scatter_mean_out_accumulates_then_divides(device):
    """When ``out`` is provided, the numerator accumulates into it before the
    final divide by per-bucket counts.
    """
    src = torch.randn(6, 4, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    out_init = torch.randn(4, 4, device=device)
    out = out_init.clone()
    result = pyg_lib.ops.scatter_mean(src, index, dim=0, out=out)

    expected = _scatter_mean_ref(src, index, dim=0, out=out_init)
    torch.testing.assert_close(result, expected)
    # The op should also have updated ``out`` in-place.
    torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_mean_empty_input(device):
    """An empty source with an empty index should yield an all-zero output
    (every bucket is empty -> count masked to 1 -> 0/1 = 0).
    """
    src = torch.empty(0, 4, device=device)
    index = torch.empty(0, dtype=torch.long, device=device)

    out = pyg_lib.ops.scatter_mean(src, index, dim=0, dim_size=3)
    assert out.size() == (3, 4)
    torch.testing.assert_close(out, torch.zeros(3, 4, device=device))


@withCUDA
def test_scatter_mean_backward_gradcheck(device):
    src = torch.randn(
        6,
        3,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    def fn(s):
        return pyg_lib.ops.scatter_mean(s, index, dim=0)

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_scatter_mean_backward_gradcheck_dim_middle(device):
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
        return pyg_lib.ops.scatter_mean(s, index, dim=1)

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_scatter_mean_backward_gradcheck_with_dim_size(device):
    """Gradcheck with an explicit (larger) ``dim_size`` exercising empty
    buckets in the output. Empty buckets have zero gradient w.r.t. ``src``.
    """
    src = torch.randn(6, dtype=torch.double, device=device, requires_grad=True)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    def fn(s):
        return pyg_lib.ops.scatter_mean(s, index, dim=-1, dim_size=6)

    assert torch.autograd.gradcheck(fn, (src,))


# ---------------------------------------------------------------------------
# scatter_min
# ---------------------------------------------------------------------------


def _scatter_min_ref(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: int = None,
):
    """Pure-PyTorch reference implementation of :func:`scatter_min`.

    Returns a tuple ``(value, argindex)``:
      * ``value[j]`` is the minimum of ``src`` entries whose broadcast index
        equals ``j`` along ``dim``.
      * ``argindex[j]`` is the position along ``dim`` of *any* entry that
        attains that minimum. On CPU the upstream contract is first-match;
        we match that here.
      * Empty buckets get ``value == 0`` and ``argindex == src.size(dim)``
        (the upstream sentinel).
    """
    if dim < 0:
        dim = dim + src.dim()
    bcast_index = _broadcast_index(index, src, dim)

    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0

    sentinel = src.size(dim)
    out_size = list(src.size())
    out_size[dim] = dim_size
    value = torch.zeros(out_size, dtype=src.dtype, device=src.device)
    argindex = torch.full(
        out_size,
        sentinel,
        dtype=torch.long,
        device=src.device,
    )

    # Walk over ``src`` along ``dim`` and, for each output position, track the
    # running min and the position of its first-match.
    # We implement this with a per-element Python loop for clarity.
    other_shape = list(src.size())
    del other_shape[dim]
    for i in range(src.size(dim)):
        src_slice = src.select(dim, i)
        idx_slice = bcast_index.select(dim, i)
        flat_src = src_slice.reshape(-1)
        flat_idx = idx_slice.reshape(-1)
        for k in range(flat_idx.numel()):
            j = int(flat_idx[k].item())
            if j < 0 or j >= dim_size:
                continue
            # Recover the multi-index for the "other" dims.
            coord = []
            rem = k
            for s in reversed(other_shape):
                coord.append(rem % s)
                rem //= s
            coord = list(reversed(coord))
            out_idx = list(coord)
            out_idx.insert(dim, j)
            out_idx_t = tuple(out_idx)
            if argindex[out_idx_t].item() == sentinel:
                # First contribution into this bucket; always take it.
                value[out_idx_t] = flat_src[k]
                argindex[out_idx_t] = i
            else:
                cur = value[out_idx_t]
                v = flat_src[k]
                if bool(v < cur):  # strict less: first-match tie-break
                    value[out_idx_t] = v
                    argindex[out_idx_t] = i

    # Empty buckets: value == 0 (already initialized), argindex stays at
    # sentinel == src.size(dim).
    return value, argindex


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
def test_scatter_min_forward_dtypes(dtype, device):
    """Forward correctness on a unique-value fixture across dtypes.

    Unique values mean argindex is unambiguous on both CPU and CUDA.
    """
    if dtype in (torch.int32, torch.int64):
        # A unique-value integer fixture covering the chosen buckets.
        src = torch.tensor(
            [[9, 1, 8, 2],
             [3, 7, 0, 5],
             [4, 6, -1, 11],
             [-3, 10, 12, -2],
             [13, -4, 14, 15],
             [16, 17, 18, 19],
             [20, 21, 22, 23],
             [24, 25, 26, 27]],
            dtype=dtype,
            device=device,
        )  # yapf: disable
    else:
        # Build a unique-valued float tensor via a randperm permutation.
        torch.manual_seed(0)
        flat = (torch.randperm(32, device=device) - 16).to(dtype)
        src = flat.view(8, 4)
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0], device=device)

    value, arg = pyg_lib.ops.scatter_min(src, index, dim=0)
    ref_value, ref_arg = _scatter_min_ref(src, index, dim=0)
    if dtype in (torch.float16, torch.bfloat16):
        torch.testing.assert_close(value, ref_value, atol=1e-2, rtol=1e-2)
    else:
        torch.testing.assert_close(value, ref_value)
    # Unique values -> exact argindex equivalence on both CPU and CUDA.
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_scatter_min_forward_dim_neg1(device):
    # Unique values along dim=-1 ensure deterministic argindex.
    torch.manual_seed(0)
    src = (torch.randperm(3 * 8, device=device).to(torch.float32) - 12).view(
        3,
        8,
    )
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0], device=device)

    value, arg = pyg_lib.ops.scatter_min(src, index, dim=-1)
    ref_value, ref_arg = _scatter_min_ref(src, index, dim=-1)
    assert value.size() == (3, 4)
    assert arg.size() == (3, 4)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_scatter_min_forward_dim_nonneg(device):
    torch.manual_seed(0)
    src = (torch.randperm(8 * 4, device=device).to(torch.float32) - 16).view(
        8,
        4,
    )
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0], device=device)

    value, arg = pyg_lib.ops.scatter_min(src, index, dim=0)
    ref_value, ref_arg = _scatter_min_ref(src, index, dim=0)
    assert value.size() == (4, 4)
    assert arg.size() == (4, 4)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_scatter_min_forward_dim_middle(device):
    torch.manual_seed(0)
    src = (
        torch.randperm(3 * 6 * 5, device=device).to(torch.float32) - 40
    ).view(3, 6, 5)
    index = torch.tensor([0, 2, 1, 0, 2, 1], device=device)

    value, arg = pyg_lib.ops.scatter_min(src, index, dim=1)
    ref_value, ref_arg = _scatter_min_ref(src, index, dim=1)
    assert value.size() == (3, 3, 5)
    assert arg.size() == (3, 3, 5)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_scatter_min_broadcasting_1d_index(device):
    """1-D ``index`` broadcasts to 2-D ``src`` along ``dim``."""
    torch.manual_seed(0)
    src = (torch.randperm(6 * 4, device=device).to(torch.float32) - 12).view(
        6,
        4,
    )
    index = torch.tensor([0, 1, 0, 2, 1, 2], device=device)

    value, arg = pyg_lib.ops.scatter_min(src, index, dim=0)
    ref_value, ref_arg = _scatter_min_ref(src, index, dim=0)
    assert value.size() == (3, 4)
    assert arg.size() == (3, 4)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_scatter_min_dim_size_auto_infer(device):
    src = torch.tensor([5.0, -1.0, 3.0, -7.0, 2.0, 9.0], device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    value, arg = pyg_lib.ops.scatter_min(src, index, dim=-1)
    # Auto-inferred dim_size = index.max() + 1 = 4.
    assert value.size(-1) == 4
    assert arg.size(-1) == 4
    ref_value, ref_arg = _scatter_min_ref(src, index, dim=-1)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_scatter_min_dim_size_explicit_empty_buckets(device):
    """Explicit ``dim_size`` larger than implicit — trailing buckets are
    empty. Upstream convention: value == 0, argindex == sentinel
    (== src.size(dim)).
    """
    src = torch.tensor([5.0, -1.0, 3.0, -7.0, 2.0, 9.0], device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)
    sentinel = src.size(-1)  # 6

    value, arg = pyg_lib.ops.scatter_min(
        src,
        index,
        dim=-1,
        dim_size=6,
    )
    assert value.size(-1) == 6
    assert arg.size(-1) == 6
    ref_value, ref_arg = _scatter_min_ref(src, index, dim=-1, dim_size=6)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)
    # Empty trailing buckets at positions 2, 4, 5 must have value 0 and
    # argindex == sentinel.
    empty_positions = [2, 4, 5]
    for p in empty_positions:
        assert value[p].item() == 0
        assert arg[p].item() == sentinel


@withCUDA
def test_scatter_min_empty_bucket_in_middle(device):
    """Empty buckets anywhere (not just trailing) yield value 0 + sentinel
    argindex.
    """
    # index skips bucket 2 entirely.
    src = torch.tensor(
        [[1.0, 2.0, 3.0],
         [-1.0, 4.0, 5.0],
         [6.0, -3.0, 7.0],
         [8.0, 9.0, -10.0],
         [11.0, 12.0, 13.0],
         [14.0, 15.0, 16.0]],
        device=device,
    )  # yapf: disable
    index = torch.tensor([0, 1, 0, 1, 3, 3], device=device)
    sentinel = src.size(0)  # 6

    value, arg = pyg_lib.ops.scatter_min(src, index, dim=0, dim_size=4)
    ref_value, ref_arg = _scatter_min_ref(src, index, dim=0, dim_size=4)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)
    # Row 2 has no contributors -> zero row + sentinel arg row.
    torch.testing.assert_close(
        value[2],
        torch.zeros(3, device=device),
    )
    torch.testing.assert_close(
        arg[2],
        torch.full((3,), sentinel, dtype=torch.long, device=device),
    )


@withCUDA
def test_scatter_min_out_overrides_init_buffer(device):
    """When ``out`` is provided, the op writes the per-bucket min into it
    (per the contract: out is initialized to numeric_limits::max() internally,
    callers supplying ``out`` are responsible for any non-default initial
    state).
    """
    src = torch.tensor([5.0, -1.0, 3.0, -7.0, 2.0, 9.0], device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    # Pre-fill out with something that should NOT contaminate the min — the
    # per-bucket min of src is strictly less than these values for every
    # non-empty bucket.
    out = torch.full((4,), 100.0, device=device)
    result_value, result_arg = pyg_lib.ops.scatter_min(
        src,
        index,
        dim=0,
        out=out,
    )

    # The non-empty buckets should match the reference exactly.
    ref_value, ref_arg = _scatter_min_ref(src, index, dim=0, dim_size=4)
    # Non-empty buckets: 0 (idx 0,2), 1 (idx 1,3,4), 3 (idx 5).
    # Bucket 2 is empty; with out=, the upstream contract leaves the caller's
    # value in place when nothing is written. We only assert the non-empty
    # buckets match the reference.
    non_empty = [0, 1, 3]
    for p in non_empty:
        torch.testing.assert_close(result_value[p], ref_value[p])
        torch.testing.assert_close(result_arg[p], ref_arg[p])


@withCUDA
def test_scatter_min_empty_input(device):
    """An empty source with an empty index yields an all-zero value tensor
    and an all-sentinel argindex tensor.
    """
    src = torch.empty(0, 4, device=device)
    index = torch.empty(0, dtype=torch.long, device=device)
    sentinel = src.size(0)  # 0

    value, arg = pyg_lib.ops.scatter_min(src, index, dim=0, dim_size=3)
    assert value.size() == (3, 4)
    assert arg.size() == (3, 4)
    torch.testing.assert_close(value, torch.zeros(3, 4, device=device))
    torch.testing.assert_close(
        arg,
        torch.full((3, 4), sentinel, dtype=torch.long, device=device),
    )


@withCUDA
@withSeed
def test_scatter_min_argindex_ties_returns_valid(device):
    """Tied values: only assert the returned argindex is *a valid* argmin,
    not a specific tie-breaker. CUDA atomic ordering is non-deterministic.
    """
    # Construct a deliberate tie: indices 0 and 2 both map to bucket 0 with
    # value 1.0; indices 1 and 4 both map to bucket 1 with value 2.0.
    src = torch.tensor(
        [1.0, 2.0, 1.0, 5.0, 2.0, 7.0],
        device=device,
    )
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    value, arg = pyg_lib.ops.scatter_min(src, index, dim=-1)
    # Value tensor must equal the true per-bucket min regardless of tie-break.
    expected_value = torch.tensor(
        [1.0, 2.0, 0.0, 7.0],
        device=device,
    )
    torch.testing.assert_close(value, expected_value)
    # Argindex on bucket 0 must be 0 or 2 (both have value 1.0).
    assert int(arg[0].item()) in (0, 2)
    # Argindex on bucket 1 must be one of the value-2.0 positions: 1 or 4.
    assert int(arg[1].item()) in (1, 4)
    # Bucket 2 is empty -> sentinel.
    assert int(arg[2].item()) == src.size(0)
    # Bucket 3 has a unique value at position 5.
    assert int(arg[3].item()) == 5
    # All non-sentinel argindexes must in fact attain the bucket's min value.
    for j in range(value.size(0)):
        a = int(arg[j].item())
        if a == src.size(0):
            continue  # empty bucket
        assert src[a].item() == value[j].item(), (
            f'arg[{j}]={a} points to src value {src[a].item()} but bucket '
            f'min is {value[j].item()}'
        )


@withCUDA
def test_scatter_min_arg_non_differentiable(device):
    """The argindex output must have ``requires_grad=False`` even when the
    value output participates in autograd.
    """
    src = torch.randn(6, dtype=torch.double, device=device, requires_grad=True)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    value, arg = pyg_lib.ops.scatter_min(src, index, dim=-1)
    # Value tensor must require grad (forwarded from src).
    assert value.requires_grad
    # Argindex must NOT require grad — it's non-differentiable.
    assert not arg.requires_grad
    # And it must be an integer tensor.
    assert arg.dtype in (torch.long, torch.int64)


@withCUDA
def test_scatter_min_backward_gradcheck(device):
    """Gradcheck on the value output. Argindex is non-differentiable and
    excluded by extracting ``[0]`` from the tuple.

    Uses unique-valued src so the active argindex is deterministic and the
    finite-difference numerical Jacobian aligns with the analytical one.
    """
    # Unique-valued src via randperm avoids ties (which would break gradcheck).
    torch.manual_seed(0)
    src = (
        (torch.randperm(6 * 3, device=device).to(torch.double) - 9)
        .view(6, 3)
        .requires_grad_(True)
    )
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    def fn(s):
        return pyg_lib.ops.scatter_min(s, index, dim=0)[0]

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_scatter_min_backward_gradcheck_dim_middle(device):
    torch.manual_seed(0)
    src = (
        (torch.randperm(2 * 6 * 3, device=device).to(torch.double) - 18)
        .view(2, 6, 3)
        .requires_grad_(True)
    )
    index = torch.tensor([0, 1, 0, 1, 2, 1], device=device)

    def fn(s):
        return pyg_lib.ops.scatter_min(s, index, dim=1)[0]

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_scatter_min_backward_gradcheck_with_dim_size(device):
    """Gradcheck with an explicit (larger) ``dim_size`` exercising empty
    buckets. Their argindex points at the sentinel (== src.size(dim)) and
    the ``+1``/``narrow`` backward pattern in the implementation drops that
    slot — so the gradient w.r.t. ``src`` for empty buckets is zero.
    """
    torch.manual_seed(0)
    src = (
        torch.randperm(6, device=device).to(torch.double) - 3
    ).requires_grad_(True)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    def fn(s):
        return pyg_lib.ops.scatter_min(s, index, dim=-1, dim_size=6)[0]

    assert torch.autograd.gradcheck(fn, (src,))


# ---------------------------------------------------------------------------
# scatter_max
# ---------------------------------------------------------------------------


def _scatter_max_ref(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: int = None,
):
    """Pure-PyTorch reference implementation of :func:`scatter_max`.

    Returns a tuple ``(value, argindex)``:
      * ``value[j]`` is the maximum of ``src`` entries whose broadcast index
        equals ``j`` along ``dim``.
      * ``argindex[j]`` is the position along ``dim`` of *any* entry that
        attains that maximum. On CPU the upstream contract is first-match;
        we match that here.
      * Empty buckets get ``value == 0`` and ``argindex == src.size(dim)``
        (the upstream sentinel).
    """
    if dim < 0:
        dim = dim + src.dim()
    bcast_index = _broadcast_index(index, src, dim)

    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0

    sentinel = src.size(dim)
    out_size = list(src.size())
    out_size[dim] = dim_size
    value = torch.zeros(out_size, dtype=src.dtype, device=src.device)
    argindex = torch.full(
        out_size,
        sentinel,
        dtype=torch.long,
        device=src.device,
    )

    # Walk over ``src`` along ``dim`` and, for each output position, track the
    # running max and the position of its first-match.
    # We implement this with a per-element Python loop for clarity.
    other_shape = list(src.size())
    del other_shape[dim]
    for i in range(src.size(dim)):
        src_slice = src.select(dim, i)
        idx_slice = bcast_index.select(dim, i)
        flat_src = src_slice.reshape(-1)
        flat_idx = idx_slice.reshape(-1)
        for k in range(flat_idx.numel()):
            j = int(flat_idx[k].item())
            if j < 0 or j >= dim_size:
                continue
            # Recover the multi-index for the "other" dims.
            coord = []
            rem = k
            for s in reversed(other_shape):
                coord.append(rem % s)
                rem //= s
            coord = list(reversed(coord))
            out_idx = list(coord)
            out_idx.insert(dim, j)
            out_idx_t = tuple(out_idx)
            if argindex[out_idx_t].item() == sentinel:
                # First contribution into this bucket; always take it.
                value[out_idx_t] = flat_src[k]
                argindex[out_idx_t] = i
            else:
                cur = value[out_idx_t]
                v = flat_src[k]
                if bool(v > cur):  # strict greater: first-match tie-break
                    value[out_idx_t] = v
                    argindex[out_idx_t] = i

    # Empty buckets: value == 0 (already initialized), argindex stays at
    # sentinel == src.size(dim).
    return value, argindex


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
def test_scatter_max_forward_dtypes(dtype, device):
    """Forward correctness on a unique-value fixture across dtypes.

    Unique values mean argindex is unambiguous on both CPU and CUDA.
    """
    if dtype in (torch.int32, torch.int64):
        # A unique-value integer fixture covering the chosen buckets.
        src = torch.tensor(
            [[9, 1, 8, 2],
             [3, 7, 0, 5],
             [4, 6, -1, 11],
             [-3, 10, 12, -2],
             [13, -4, 14, 15],
             [16, 17, 18, 19],
             [20, 21, 22, 23],
             [24, 25, 26, 27]],
            dtype=dtype,
            device=device,
        )  # yapf: disable
    else:
        # Build a unique-valued float tensor via a randperm permutation.
        torch.manual_seed(0)
        flat = (torch.randperm(32, device=device) - 16).to(dtype)
        src = flat.view(8, 4)
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0], device=device)

    value, arg = pyg_lib.ops.scatter_max(src, index, dim=0)
    ref_value, ref_arg = _scatter_max_ref(src, index, dim=0)
    if dtype in (torch.float16, torch.bfloat16):
        torch.testing.assert_close(value, ref_value, atol=1e-2, rtol=1e-2)
    else:
        torch.testing.assert_close(value, ref_value)
    # Unique values -> exact argindex equivalence on both CPU and CUDA.
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_scatter_max_forward_dim_neg1(device):
    # Unique values along dim=-1 ensure deterministic argindex.
    torch.manual_seed(0)
    src = (torch.randperm(3 * 8, device=device).to(torch.float32) - 12).view(
        3,
        8,
    )
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0], device=device)

    value, arg = pyg_lib.ops.scatter_max(src, index, dim=-1)
    ref_value, ref_arg = _scatter_max_ref(src, index, dim=-1)
    assert value.size() == (3, 4)
    assert arg.size() == (3, 4)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_scatter_max_forward_dim_nonneg(device):
    torch.manual_seed(0)
    src = (torch.randperm(8 * 4, device=device).to(torch.float32) - 16).view(
        8,
        4,
    )
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0], device=device)

    value, arg = pyg_lib.ops.scatter_max(src, index, dim=0)
    ref_value, ref_arg = _scatter_max_ref(src, index, dim=0)
    assert value.size() == (4, 4)
    assert arg.size() == (4, 4)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_scatter_max_forward_dim_middle(device):
    torch.manual_seed(0)
    src = (
        torch.randperm(3 * 6 * 5, device=device).to(torch.float32) - 40
    ).view(3, 6, 5)
    index = torch.tensor([0, 2, 1, 0, 2, 1], device=device)

    value, arg = pyg_lib.ops.scatter_max(src, index, dim=1)
    ref_value, ref_arg = _scatter_max_ref(src, index, dim=1)
    assert value.size() == (3, 3, 5)
    assert arg.size() == (3, 3, 5)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_scatter_max_broadcasting_1d_index(device):
    """1-D ``index`` broadcasts to 2-D ``src`` along ``dim``."""
    torch.manual_seed(0)
    src = (torch.randperm(6 * 4, device=device).to(torch.float32) - 12).view(
        6,
        4,
    )
    index = torch.tensor([0, 1, 0, 2, 1, 2], device=device)

    value, arg = pyg_lib.ops.scatter_max(src, index, dim=0)
    ref_value, ref_arg = _scatter_max_ref(src, index, dim=0)
    assert value.size() == (3, 4)
    assert arg.size() == (3, 4)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_scatter_max_dim_size_auto_infer(device):
    src = torch.tensor([5.0, -1.0, 3.0, -7.0, 2.0, 9.0], device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    value, arg = pyg_lib.ops.scatter_max(src, index, dim=-1)
    # Auto-inferred dim_size = index.max() + 1 = 4.
    assert value.size(-1) == 4
    assert arg.size(-1) == 4
    ref_value, ref_arg = _scatter_max_ref(src, index, dim=-1)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_scatter_max_dim_size_explicit_empty_buckets(device):
    """Explicit ``dim_size`` larger than implicit — trailing buckets are
    empty. Upstream convention: value == 0, argindex == sentinel
    (== src.size(dim)).
    """
    src = torch.tensor([5.0, -1.0, 3.0, -7.0, 2.0, 9.0], device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)
    sentinel = src.size(-1)  # 6

    value, arg = pyg_lib.ops.scatter_max(
        src,
        index,
        dim=-1,
        dim_size=6,
    )
    assert value.size(-1) == 6
    assert arg.size(-1) == 6
    ref_value, ref_arg = _scatter_max_ref(src, index, dim=-1, dim_size=6)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)
    # Empty trailing buckets at positions 2, 4, 5 must have value 0 and
    # argindex == sentinel.
    empty_positions = [2, 4, 5]
    for p in empty_positions:
        assert value[p].item() == 0
        assert arg[p].item() == sentinel


@withCUDA
def test_scatter_max_empty_bucket_in_middle(device):
    """Empty buckets anywhere (not just trailing) yield value 0 + sentinel
    argindex.
    """
    # index skips bucket 2 entirely.
    src = torch.tensor(
        [[1.0, 2.0, 3.0],
         [-1.0, 4.0, 5.0],
         [6.0, -3.0, 7.0],
         [8.0, 9.0, -10.0],
         [11.0, 12.0, 13.0],
         [14.0, 15.0, 16.0]],
        device=device,
    )  # yapf: disable
    index = torch.tensor([0, 1, 0, 1, 3, 3], device=device)
    sentinel = src.size(0)  # 6

    value, arg = pyg_lib.ops.scatter_max(src, index, dim=0, dim_size=4)
    ref_value, ref_arg = _scatter_max_ref(src, index, dim=0, dim_size=4)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)
    # Row 2 has no contributors -> zero row + sentinel arg row.
    torch.testing.assert_close(
        value[2],
        torch.zeros(3, device=device),
    )
    torch.testing.assert_close(
        arg[2],
        torch.full((3,), sentinel, dtype=torch.long, device=device),
    )


@withCUDA
def test_scatter_max_out_overrides_init_buffer(device):
    """When ``out`` is provided, the op writes the per-bucket max into it
    (per the contract: out is initialized to numeric_limits::lowest()
    internally; callers supplying ``out`` are responsible for any non-default
    initial state).
    """
    src = torch.tensor([5.0, -1.0, 3.0, -7.0, 2.0, 9.0], device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    # Pre-fill out with something that should NOT contaminate the max — the
    # per-bucket max of src is strictly greater than these values for every
    # non-empty bucket.
    out = torch.full((4,), -100.0, device=device)
    result_value, result_arg = pyg_lib.ops.scatter_max(
        src,
        index,
        dim=0,
        out=out,
    )

    # The non-empty buckets should match the reference exactly.
    ref_value, ref_arg = _scatter_max_ref(src, index, dim=0, dim_size=4)
    # Non-empty buckets: 0 (idx 0,2), 1 (idx 1,3,4), 3 (idx 5).
    # Bucket 2 is empty; with out=, the upstream contract leaves the caller's
    # value in place when nothing is written. We only assert the non-empty
    # buckets match the reference.
    non_empty = [0, 1, 3]
    for p in non_empty:
        torch.testing.assert_close(result_value[p], ref_value[p])
        torch.testing.assert_close(result_arg[p], ref_arg[p])


@withCUDA
def test_scatter_max_empty_input(device):
    """An empty source with an empty index yields an all-zero value tensor
    and an all-sentinel argindex tensor.
    """
    src = torch.empty(0, 4, device=device)
    index = torch.empty(0, dtype=torch.long, device=device)
    sentinel = src.size(0)  # 0

    value, arg = pyg_lib.ops.scatter_max(src, index, dim=0, dim_size=3)
    assert value.size() == (3, 4)
    assert arg.size() == (3, 4)
    torch.testing.assert_close(value, torch.zeros(3, 4, device=device))
    torch.testing.assert_close(
        arg,
        torch.full((3, 4), sentinel, dtype=torch.long, device=device),
    )


@withCUDA
@withSeed
def test_scatter_max_argindex_ties_returns_valid(device):
    """Tied values: only assert the returned argindex is *a valid* argmax,
    not a specific tie-breaker. CUDA atomic ordering is non-deterministic.
    """
    # Construct a deliberate tie: indices 0 and 2 both map to bucket 0 with
    # value 1.0; indices 1 and 4 both map to bucket 1 with value 2.0.
    src = torch.tensor(
        [1.0, 2.0, 1.0, -5.0, 2.0, -7.0],
        device=device,
    )
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    value, arg = pyg_lib.ops.scatter_max(src, index, dim=-1)
    # Value tensor must equal the true per-bucket max regardless of tie-break.
    expected_value = torch.tensor(
        [1.0, 2.0, 0.0, -7.0],
        device=device,
    )
    torch.testing.assert_close(value, expected_value)
    # Argindex on bucket 0 must be 0 or 2 (both have value 1.0).
    assert int(arg[0].item()) in (0, 2)
    # Argindex on bucket 1 must be one of the value-2.0 positions: 1 or 4.
    assert int(arg[1].item()) in (1, 4)
    # Bucket 2 is empty -> sentinel.
    assert int(arg[2].item()) == src.size(0)
    # Bucket 3 has a unique value at position 5.
    assert int(arg[3].item()) == 5
    # All non-sentinel argindexes must in fact attain the bucket's max value.
    for j in range(value.size(0)):
        a = int(arg[j].item())
        if a == src.size(0):
            continue  # empty bucket
        assert src[a].item() == value[j].item(), (
            f'arg[{j}]={a} points to src value {src[a].item()} but bucket '
            f'max is {value[j].item()}'
        )


@withCUDA
def test_scatter_max_arg_non_differentiable(device):
    """The argindex output must have ``requires_grad=False`` even when the
    value output participates in autograd.
    """
    src = torch.randn(6, dtype=torch.double, device=device, requires_grad=True)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    value, arg = pyg_lib.ops.scatter_max(src, index, dim=-1)
    # Value tensor must require grad (forwarded from src).
    assert value.requires_grad
    # Argindex must NOT require grad — it's non-differentiable.
    assert not arg.requires_grad
    # And it must be an integer tensor.
    assert arg.dtype in (torch.long, torch.int64)


@withCUDA
def test_scatter_max_backward_gradcheck(device):
    """Gradcheck on the value output. Argindex is non-differentiable and
    excluded by extracting ``[0]`` from the tuple.

    Uses unique-valued src so the active argindex is deterministic and the
    finite-difference numerical Jacobian aligns with the analytical one.
    """
    # Unique-valued src via randperm avoids ties (which would break gradcheck).
    torch.manual_seed(0)
    src = (
        (torch.randperm(6 * 3, device=device).to(torch.double) - 9)
        .view(6, 3)
        .requires_grad_(True)
    )
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    def fn(s):
        return pyg_lib.ops.scatter_max(s, index, dim=0)[0]

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_scatter_max_backward_gradcheck_dim_middle(device):
    torch.manual_seed(0)
    src = (
        (torch.randperm(2 * 6 * 3, device=device).to(torch.double) - 18)
        .view(2, 6, 3)
        .requires_grad_(True)
    )
    index = torch.tensor([0, 1, 0, 1, 2, 1], device=device)

    def fn(s):
        return pyg_lib.ops.scatter_max(s, index, dim=1)[0]

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_scatter_max_backward_gradcheck_with_dim_size(device):
    """Gradcheck with an explicit (larger) ``dim_size`` exercising empty
    buckets. Their argindex points at the sentinel (== src.size(dim)) and
    the ``+1``/``narrow`` backward pattern in the implementation drops that
    slot — so the gradient w.r.t. ``src`` for empty buckets is zero.
    """
    torch.manual_seed(0)
    src = (
        torch.randperm(6, device=device).to(torch.double) - 3
    ).requires_grad_(True)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    def fn(s):
        return pyg_lib.ops.scatter_max(s, index, dim=-1, dim_size=6)[0]

    assert torch.autograd.gradcheck(fn, (src,))


# ---------------------------------------------------------------------------
# scatter dispatcher (commit 14 — Python layer)
# ---------------------------------------------------------------------------


@withCUDA
@pytest.mark.parametrize(
    'reduce',
    ['sum', 'add', 'mul', 'mean', 'min', 'max'],
)
def test_scatter_dispatcher(reduce, device):
    """``scatter(src, index, dim, out, dim_size, reduce=...)`` must route to
    the corresponding typed op. For ``min``/``max`` the dispatcher returns
    ``[0]`` (value only), not the ``(value, argindex)`` tuple.

    Uses a unique-valued ``src`` so argindex tie-breaks are deterministic
    and ``min``/``max`` value outputs compare exactly across devices.
    """
    torch.manual_seed(0)
    src = (torch.randperm(6 * 3, device=device).to(torch.float64) - 9).view(
        6,
        3,
    )
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    out = pyg_lib.ops.scatter(src, index, dim=0, dim_size=4, reduce=reduce)
    if reduce in ('sum', 'add'):
        expected = pyg_lib.ops.scatter_sum(src, index, dim=0, dim_size=4)
    elif reduce == 'mul':
        expected = pyg_lib.ops.scatter_mul(src, index, dim=0, dim_size=4)
    elif reduce == 'mean':
        expected = pyg_lib.ops.scatter_mean(src, index, dim=0, dim_size=4)
    elif reduce == 'min':
        expected = pyg_lib.ops.scatter_min(src, index, dim=0, dim_size=4)[0]
    elif reduce == 'max':
        expected = pyg_lib.ops.scatter_max(src, index, dim=0, dim_size=4)[0]
    assert isinstance(out, torch.Tensor)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_dispatcher_unknown_reduce_raises(device):
    """The dispatcher must reject unknown reduce strings with a clear error."""
    src = torch.randn(6, 3, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)
    with pytest.raises(ValueError):
        pyg_lib.ops.scatter(src, index, dim=0, reduce='unsupported')


@withCUDA
def test_scatter_dispatcher_default_reduce_is_sum(device):
    """Default ``reduce`` is ``"sum"`` (upstream convention)."""
    torch.manual_seed(0)
    src = torch.randn(6, 3, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3], device=device)

    out = pyg_lib.ops.scatter(src, index, dim=0)
    expected = pyg_lib.ops.scatter_sum(src, index, dim=0)
    torch.testing.assert_close(out, expected)
