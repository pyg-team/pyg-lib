import math

import pytest
import torch

import pyg_lib
from pyg_lib.testing import withCUDA, withSeed

# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _broadcast_index(
    index: torch.Tensor,
    src: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Broadcasts a 1-D :obj:`index` to the shape of :obj:`src` along
    :obj:`dim`. Mirrors the helper used elsewhere in the test suite.
    """
    if index.dim() == src.dim():
        return index
    if dim < 0:
        dim = dim + src.dim()
    size = [1] * src.dim()
    size[dim] = -1
    return index.view(size).expand_as(src)


def _scatter_softmax_ref(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: int = None,
) -> torch.Tensor:
    """Per-bucket softmax reference: max-recenter, exp, divide by per-bucket
    sum. Reproduces the upstream ``torch_scatter.composite.scatter_softmax``
    algorithm verbatim using plain PyTorch primitives.
    """
    if dim < 0:
        dim = dim + src.dim()
    bcast_index = _broadcast_index(index, src, dim)
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    size = list(src.size())
    size[dim] = dim_size
    # Per-bucket max via ``scatter_reduce_(..., amax, include_self=False)``.
    max_per_idx = src.new_full(size, float('-inf')).scatter_reduce_(
        dim,
        bcast_index,
        src,
        reduce='amax',
        include_self=False,
    )
    max_per_src = max_per_idx.gather(dim, bcast_index)
    recentered = (src - max_per_src).exp()
    sum_per_idx = src.new_zeros(size).scatter_add_(
        dim,
        bcast_index,
        recentered,
    )
    return recentered / sum_per_idx.gather(dim, bcast_index)


def _scatter_log_softmax_ref(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: int = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-bucket log-softmax reference: ``(src - max) - log(sum + eps)``."""
    if dim < 0:
        dim = dim + src.dim()
    bcast_index = _broadcast_index(index, src, dim)
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    size = list(src.size())
    size[dim] = dim_size
    max_per_idx = src.new_full(size, float('-inf')).scatter_reduce_(
        dim,
        bcast_index,
        src,
        reduce='amax',
        include_self=False,
    )
    max_per_src = max_per_idx.gather(dim, bcast_index)
    recentered = src - max_per_src
    sum_per_idx = src.new_zeros(size).scatter_add_(
        dim,
        bcast_index,
        recentered.exp(),
    )
    return recentered - (sum_per_idx + eps).log().gather(dim, bcast_index)


def _scatter_std_ref(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: int = None,
    unbiased: bool = True,
) -> torch.Tensor:
    """Per-bucket std reference matching upstream's two-pass scatter_sum
    algorithm (used to avoid integer/float coupling). Bessel correction
    ``N / (N - 1)`` applied when ``unbiased=True``.
    """
    if dim < 0:
        dim = dim + src.dim()
    bcast_index = _broadcast_index(index, src, dim)
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    size = list(src.size())
    size[dim] = dim_size
    ones = torch.ones_like(src)
    count = src.new_zeros(size).scatter_add_(dim, bcast_index, ones)
    total = src.new_zeros(size).scatter_add_(dim, bcast_index, src)
    mean = total / count.clamp(min=1)
    diff = src - mean.gather(dim, bcast_index)
    sq_sum = src.new_zeros(size).scatter_add_(dim, bcast_index, diff * diff)
    denom = count.sub(1).clamp_(min=1) if unbiased else count.clamp(min=1)
    return (sq_sum / (denom + 1e-6)).sqrt()


def _scatter_logsumexp_ref(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: int = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Numerically-stable per-bucket logsumexp reference:
    ``max + log(sum(exp(src - max)) + eps)``. Empty buckets yield 0.
    """
    if dim < 0:
        dim = dim + src.dim()
    bcast_index = _broadcast_index(index, src, dim)
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    size = list(src.size())
    size[dim] = dim_size
    max_per_idx = src.new_full(size, float('-inf')).scatter_reduce_(
        dim,
        bcast_index,
        src,
        reduce='amax',
        include_self=False,
    )
    max_per_src = max_per_idx.gather(dim, bcast_index)
    recentered = src - max_per_src
    recentered = torch.where(
        torch.isnan(recentered),
        torch.full_like(recentered, float('-inf')),
        recentered,
    )
    sum_per_idx = src.new_zeros(size).scatter_add_(
        dim,
        bcast_index,
        recentered.exp(),
    )
    out = max_per_idx + (sum_per_idx + eps).log()
    # Empty buckets: max == -inf -> nan_to_num to 0.
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# scatter_softmax — forward + grad
# ---------------------------------------------------------------------------


@withCUDA
@withSeed
def test_scatter_softmax_forward(device):
    """Forward correctness against a pure-PyTorch max-recentered softmax."""
    # Distinct random values per bucket avoid argmax ties.
    src = torch.randn(8, 4, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0], device=device)

    out = pyg_lib.ops.scatter_softmax(src, index, dim=0)
    expected = _scatter_softmax_ref(src, index, dim=0)
    torch.testing.assert_close(out, expected)


@withCUDA
@withSeed
def test_scatter_softmax_normalizes_per_bucket(device):
    """The per-bucket sum of the softmax must be 1 (or 0 for empty buckets)."""
    src = torch.randn(10, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0, 2, 3], device=device)

    out = pyg_lib.ops.scatter_softmax(src, index, dim=0, dim_size=4)
    sums = torch.zeros(4, dtype=out.dtype, device=device).scatter_add_(
        0,
        index,
        out,
    )
    torch.testing.assert_close(sums, torch.ones(4, device=device))


@withCUDA
@withSeed
def test_scatter_softmax_backward_gradcheck(device):
    """Gradcheck with distinct double-precision values so argmax never ties."""
    # ``torch.randperm`` gives unique integers; scale + cast for double values.
    src = (
        ((torch.randperm(6 * 3, device=device).to(torch.double) - 9) * 0.1)
        .view(6, 3)
        .requires_grad_(True)
    )
    index = torch.tensor([0, 1, 0, 1, 1, 2], device=device)

    def fn(s):
        return pyg_lib.ops.scatter_softmax(s, index, dim=0)

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_scatter_softmax_rejects_integer_input(device):
    """Integer ``src`` must raise a clear error (upstream behavior)."""
    src = torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device)
    index = torch.tensor([0, 0, 1, 1], device=device)
    with pytest.raises((ValueError, TypeError)):
        pyg_lib.ops.scatter_softmax(src, index, dim=0)


# ---------------------------------------------------------------------------
# scatter_log_softmax — forward + grad
# ---------------------------------------------------------------------------


@withCUDA
@withSeed
def test_scatter_log_softmax_forward(device):
    """Forward correctness against a pure-PyTorch log-softmax reference."""
    src = torch.randn(8, 4, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0], device=device)

    out = pyg_lib.ops.scatter_log_softmax(src, index, dim=0)
    expected = _scatter_log_softmax_ref(src, index, dim=0)
    torch.testing.assert_close(out, expected)


@withCUDA
@withSeed
def test_scatter_log_softmax_matches_log_of_softmax(device):
    """``log_softmax`` and ``log(softmax)`` should agree up to numerics."""
    src = torch.randn(8, 4, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0], device=device)

    log_sm = pyg_lib.ops.scatter_log_softmax(src, index, dim=0)
    sm = pyg_lib.ops.scatter_softmax(src, index, dim=0)
    torch.testing.assert_close(log_sm, sm.log(), atol=1e-5, rtol=1e-5)


@withCUDA
@withSeed
def test_scatter_log_softmax_backward_gradcheck(device):
    """Gradcheck with distinct double-precision values so argmax never ties."""
    src = (
        ((torch.randperm(6 * 3, device=device).to(torch.double) - 9) * 0.1)
        .view(6, 3)
        .requires_grad_(True)
    )
    index = torch.tensor([0, 1, 0, 1, 1, 2], device=device)

    def fn(s):
        return pyg_lib.ops.scatter_log_softmax(s, index, dim=0)

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_scatter_log_softmax_rejects_integer_input(device):
    """Integer ``src`` must raise a clear error (upstream behavior)."""
    src = torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device)
    index = torch.tensor([0, 0, 1, 1], device=device)
    with pytest.raises((ValueError, TypeError)):
        pyg_lib.ops.scatter_log_softmax(src, index, dim=0)


# ---------------------------------------------------------------------------
# scatter_std — forward + grad
# ---------------------------------------------------------------------------


@withCUDA
@withSeed
@pytest.mark.parametrize('unbiased', [True, False])
def test_scatter_std_forward(unbiased, device):
    """Forward correctness against a pure-PyTorch per-bucket std reference."""
    src = torch.randn(8, 4, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0], device=device)

    out = pyg_lib.ops.scatter_std(
        src,
        index,
        dim=0,
        unbiased=unbiased,
    )
    expected = _scatter_std_ref(src, index, dim=0, unbiased=unbiased)
    torch.testing.assert_close(out, expected)


@withCUDA
@withSeed
@pytest.mark.parametrize('unbiased', [True, False])
def test_scatter_std_backward_gradcheck(unbiased, device):
    """Gradcheck for the std composite. Uses distinct random values so no
    degenerate single-element bucket has zero variance gradient ambiguity.
    """
    src = (
        ((torch.randperm(8 * 3, device=device).to(torch.double) - 12) * 0.1)
        .view(8, 3)
        .requires_grad_(True)
    )
    # Every bucket gets at least two elements to avoid zero-variance cases.
    index = torch.tensor([0, 1, 0, 1, 2, 2, 0, 1], device=device)

    def fn(s):
        return pyg_lib.ops.scatter_std(s, index, dim=0, unbiased=unbiased)

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_scatter_std_rejects_integer_input(device):
    """Integer ``src`` must raise a clear error (upstream behavior)."""
    src = torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device)
    index = torch.tensor([0, 0, 1, 1], device=device)
    with pytest.raises((ValueError, TypeError)):
        pyg_lib.ops.scatter_std(src, index, dim=0)


# ---------------------------------------------------------------------------
# scatter_logsumexp — forward + numerical stability + ``out=`` path
# ---------------------------------------------------------------------------


@withCUDA
@withSeed
def test_scatter_logsumexp_forward(device):
    """Forward correctness against a pure-PyTorch logsumexp reference."""
    src = torch.randn(8, 4, device=device)
    index = torch.tensor([0, 1, 0, 1, 1, 3, 2, 0], device=device)

    out = pyg_lib.ops.scatter_logsumexp(src, index, dim=0)
    expected = _scatter_logsumexp_ref(src, index, dim=0)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_scatter_logsumexp_large_magnitude_stability(device):
    """Stability test: very large positive values would overflow naive
    ``log(sum(exp(...)))`` (``exp(1000)`` is ``inf``). The numerically-stable
    implementation must produce finite, correct results.

    Per-bucket logsumexp is ``max + log(count_exp_shifted)`` so for a bucket
    with two equal entries at value ``v`` the result is ``v + log(2)``.
    """
    # Bucket 0: two entries at 1000.0 -> expect 1000 + log(2).
    # Bucket 1: two entries at -1000.0 -> expect -1000 + log(2).
    # Bucket 2: one entry at 5.0       -> expect 5.0 exactly.
    src = torch.tensor(
        [1000.0, 1000.0, -1000.0, -1000.0, 5.0],
        device=device,
    )
    index = torch.tensor([0, 0, 1, 1, 2], device=device)

    out = pyg_lib.ops.scatter_logsumexp(src, index, dim=0)
    # All entries must be finite.
    assert torch.isfinite(out).all(), f'non-finite entries in {out}'
    expected = torch.tensor(
        [1000.0 + math.log(2.0), -1000.0 + math.log(2.0), 5.0],
        device=device,
    )
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-5)


@withCUDA
@withSeed
def test_scatter_logsumexp_empty_buckets_yield_zero(device):
    """Empty buckets must produce 0 (upstream's ``nan_to_num_(neginf=0)``)."""
    src = torch.randn(4, device=device)
    # Buckets 1 and 3 receive no entries.
    index = torch.tensor([0, 2, 0, 2], device=device)

    out = pyg_lib.ops.scatter_logsumexp(src, index, dim=0, dim_size=4)
    assert torch.isfinite(out).all()
    assert out[1].item() == 0.0
    assert out[3].item() == 0.0


@withCUDA
@withSeed
def test_scatter_logsumexp_out_restores_non_finite(device):
    """``out=...`` non-finite restoration: empty buckets in the result must
    be replaced by the corresponding pre-existing values in ``out``. Upstream
    ``scatter_logsumexp.py`` keeps ``orig_out`` and copies it into positions
    where the final result is non-finite.
    """
    src = torch.tensor([1.0, 2.0, 3.0], device=device)
    # Only bucket 0 receives entries; buckets 1, 2 are empty.
    index = torch.tensor([0, 0, 0], device=device)

    # Pre-fill ``out`` with sentinel values that must be preserved at the
    # empty-bucket positions.
    sentinel = torch.tensor([-999.0, 42.0, -7.0], device=device)
    out = sentinel.clone()
    result = pyg_lib.ops.scatter_logsumexp(src, index, dim=0, out=out)

    # Bucket 0 is non-empty -> logsumexp result (not sentinel).
    expected_0 = torch.tensor([1.0, 2.0, 3.0], device=device).logsumexp(0)
    # Loose tolerance because of the +eps inside the log.
    assert abs(result[0].item() - expected_0.item()) < 1e-4
    # Buckets 1 and 2 are empty -> sentinel values restored from original out.
    assert result[1].item() == 42.0
    assert result[2].item() == -7.0


@withCUDA
@withSeed
def test_scatter_logsumexp_backward_gradcheck(device):
    """Gradcheck on logsumexp. Uses scaled distinct values so the max within
    each bucket is unambiguous (no tied argmax that would break finite
    differencing).
    """
    src = (
        ((torch.randperm(6 * 3, device=device).to(torch.double) - 9) * 0.1)
        .view(6, 3)
        .requires_grad_(True)
    )
    index = torch.tensor([0, 1, 0, 1, 1, 2], device=device)

    def fn(s):
        return pyg_lib.ops.scatter_logsumexp(s, index, dim=0)

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_scatter_logsumexp_rejects_integer_input(device):
    """Integer ``src`` must raise a clear error (upstream behavior)."""
    src = torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device)
    index = torch.tensor([0, 0, 1, 1], device=device)
    with pytest.raises((ValueError, TypeError)):
        pyg_lib.ops.scatter_logsumexp(src, index, dim=0)
