import pytest
import torch

import pyg_lib
from pyg_lib.testing import withCUDA

# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _broadcast_index(
    index: torch.Tensor,
    src: torch.Tensor,
) -> torch.Tensor:
    """Broadcasts ``index`` up to the shape of ``src`` along the COO dim
    (``index.dim() - 1``). Mirrors the upstream / C++ ``broadcast`` util.
    """
    dim = index.dim() - 1
    if index.dim() == src.dim():
        return index
    size = [1] * src.dim()
    size[dim] = -1
    # ``index`` is 1-D in the broadcast case; reshape then expand.
    return index.view(size).expand_as(src)


def _segment_sum_coo_ref(
    src: torch.Tensor,
    index: torch.Tensor,
    out: torch.Tensor = None,
    dim_size: int = None,
) -> torch.Tensor:
    """Pure-PyTorch reference for :func:`segment_sum_coo`.

    COO ops always reduce along ``dim = index.dim() - 1``. With a sorted
    ``index``, segment-sum is equivalent to scatter-sum, so the reference is
    a plain ``zeros + scatter_add_``.
    """
    dim = index.dim() - 1
    bcast_index = _broadcast_index(index, src)
    if out is None:
        if dim_size is None:
            if index.numel() > 0:
                dim_size = int(index.max().item()) + 1
            else:
                dim_size = 0
        size = list(src.size())
        size[dim] = dim_size
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
    out.scatter_add_(dim, bcast_index, src)
    return out


def _segment_mean_coo_ref(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int = None,
) -> torch.Tensor:
    """Pure-PyTorch reference for :func:`segment_mean_coo`.

    Computes per-bucket sums via :func:`_segment_sum_coo_ref`, then divides by
    per-bucket counts. Counts for empty buckets are masked to 1 so empty
    buckets yield exactly 0 (matching upstream semantics).
    """
    dim = index.dim() - 1
    summed = _segment_sum_coo_ref(src, index, dim_size=dim_size)
    if dim_size is None:
        dim_size = summed.size(dim)
    # Count per bucket: scatter-add of ones along the COO dim.
    counts = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    if index.numel() > 0:
        counts.scatter_add_(
            0,
            index.reshape(-1),
            torch.ones(index.numel(), dtype=src.dtype, device=src.device),
        )
    # Shape counts for broadcasting along ``dim``.
    shape = [1] * summed.dim()
    shape[dim] = dim_size
    counts = counts.view(shape)
    safe_counts = counts.masked_fill(counts == 0, 1)
    return summed / safe_counts


def _gather_coo_ref(
    src: torch.Tensor,
    index: torch.Tensor,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Pure-PyTorch reference for :func:`gather_coo`.

    ``out[..., i, ...] = src[..., index[i], ...]`` along ``dim = index.dim()
    - 1``. With a 1-D ``index`` and multi-D ``src``, the index is broadcast
    along the last dim, equivalent to ``src.index_select(dim, index)``.
    """
    dim = index.dim() - 1
    if index.dim() == src.dim():
        result = torch.gather(src, dim, index)
    else:
        # 1-D index path: pick rows / cols along ``dim``.
        result = src.index_select(dim, index)
    if out is not None:
        out.copy_(result)
        return out
    return result


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------


def test_segment_add_coo_is_segment_sum_coo_alias():
    assert pyg_lib.ops.segment_add_coo is pyg_lib.ops.segment_sum_coo


# ---------------------------------------------------------------------------
# segment_sum_coo — forward
# ---------------------------------------------------------------------------


@withCUDA
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_segment_sum_coo_forward_dtypes(dtype, device):
    torch.manual_seed(0)
    src = torch.randn(8, 4, dtype=dtype, device=device)
    # Sorted ascending — required by segment_*_coo.
    index = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3], device=device)

    out = pyg_lib.ops.segment_sum_coo(src, index)
    expected = _segment_sum_coo_ref(src, index)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_sum_coo_forward_1d(device):
    src = torch.randn(8, device=device)
    index = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3], device=device)

    out = pyg_lib.ops.segment_sum_coo(src, index)
    expected = _segment_sum_coo_ref(src, index)
    assert out.size() == (4,)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_sum_coo_forward_2d_broadcast_index(device):
    """1-D ``index`` broadcasts over a 2-D ``src`` along ``dim = 0``."""
    src = torch.randn(6, 4, device=device)
    index = torch.tensor([0, 0, 1, 1, 1, 2], device=device)

    out = pyg_lib.ops.segment_sum_coo(src, index)
    expected = _segment_sum_coo_ref(src, index)
    assert out.size() == (3, 4)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_sum_coo_forward_k1_small_trailing(device):
    """K==1 path: no trailing feature dims (purely 1-D)."""
    src = torch.randn(10, device=device)
    index = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], device=device)

    out = pyg_lib.ops.segment_sum_coo(src, index)
    expected = _segment_sum_coo_ref(src, index)
    assert out.size() == (5,)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_sum_coo_forward_k_large_broadcast(device):
    """K>1 path: large trailing feature dim exercises the broadcast kernel."""
    src = torch.randn(12, 64, device=device)
    index = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4], device=device)

    out = pyg_lib.ops.segment_sum_coo(src, index)
    expected = _segment_sum_coo_ref(src, index)
    assert out.size() == (5, 64)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_sum_coo_forward_matches_scatter_sum(device):
    """Sorted ``index`` -> segment_sum_coo is equivalent to scatter_sum."""
    src = torch.randn(8, 4, device=device)
    index = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3], device=device)

    out_coo = pyg_lib.ops.segment_sum_coo(src, index)
    out_scatter = pyg_lib.ops.scatter_sum(src, index, dim=0)
    torch.testing.assert_close(out_coo, out_scatter)


@withCUDA
def test_segment_sum_coo_dim_size_auto_infer(device):
    src = torch.randn(6, device=device)
    index = torch.tensor([0, 1, 1, 2, 2, 3], device=device)

    out = pyg_lib.ops.segment_sum_coo(src, index)
    # Auto-inferred dim_size = index.max() + 1 = 4.
    assert out.size(-1) == 4
    expected = _segment_sum_coo_ref(src, index)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_sum_coo_dim_size_explicit(device):
    src = torch.randn(6, device=device)
    index = torch.tensor([0, 1, 1, 2, 2, 3], device=device)

    out = pyg_lib.ops.segment_sum_coo(src, index, dim_size=6)
    assert out.size(-1) == 6
    expected = _segment_sum_coo_ref(src, index, dim_size=6)
    torch.testing.assert_close(out, expected)
    # Trailing empty buckets should be exactly zero.
    torch.testing.assert_close(out[4:], torch.zeros(2, device=device))


@withCUDA
def test_segment_sum_coo_empty_rows_in_middle(device):
    """Row 1 has no entries — its output must be zero."""
    src = torch.randn(5, 4, device=device)
    index = torch.tensor([0, 0, 2, 2, 2], device=device)

    out = pyg_lib.ops.segment_sum_coo(src, index)
    expected = _segment_sum_coo_ref(src, index)
    assert out.size() == (3, 4)
    torch.testing.assert_close(out, expected)
    # Row 1 is empty -> zeros.
    torch.testing.assert_close(out[1], torch.zeros(4, device=device))


@withCUDA
def test_segment_sum_coo_empty_input(device):
    """Empty source and index — output is all-zero of the requested shape."""
    src = torch.empty(0, 4, device=device)
    index = torch.empty(0, dtype=torch.long, device=device)

    out = pyg_lib.ops.segment_sum_coo(src, index, dim_size=3)
    assert out.size() == (3, 4)
    torch.testing.assert_close(out, torch.zeros(3, 4, device=device))


@withCUDA
def test_segment_sum_coo_sorted_smoke(device):
    """Sorted-ascending index is the supported contract — smoke test that
    a long, well-formed sorted index produces the reference result.
    """
    torch.manual_seed(1)
    src = torch.randn(32, 3, device=device)
    # Long sorted run with varying segment sizes.
    index = torch.tensor(
        [0] * 5 + [1] * 1 + [2] * 8 + [3] * 4 + [4] * 7 + [5] * 7,
        device=device,
    )
    assert index.numel() == 32

    out = pyg_lib.ops.segment_sum_coo(src, index)
    expected = _segment_sum_coo_ref(src, index)
    torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# segment_sum_coo — backward
# ---------------------------------------------------------------------------


@withCUDA
def test_segment_sum_coo_backward_gradcheck(device):
    src = torch.randn(
        6,
        3,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    index = torch.tensor([0, 0, 1, 1, 1, 2], device=device)

    def fn(s):
        return pyg_lib.ops.segment_sum_coo(s, index)

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_segment_sum_coo_backward_gradcheck_with_dim_size(device):
    """Gradcheck with explicit (larger) ``dim_size`` -> empty trailing rows."""
    src = torch.randn(
        5,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    index = torch.tensor([0, 0, 2, 2, 2], device=device)

    def fn(s):
        return pyg_lib.ops.segment_sum_coo(s, index, dim_size=5)

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_segment_sum_coo_backward_gradcheck_empty_rows(device):
    """Empty-row fixture under gradcheck."""
    src = torch.randn(
        5,
        4,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    index = torch.tensor([0, 0, 2, 2, 2], device=device)

    def fn(s):
        return pyg_lib.ops.segment_sum_coo(s, index)

    assert torch.autograd.gradcheck(fn, (src,))


# ---------------------------------------------------------------------------
# gather_coo — forward
# ---------------------------------------------------------------------------


@withCUDA
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_gather_coo_forward_dtypes(dtype, device):
    torch.manual_seed(0)
    src = torch.randn(4, 3, dtype=dtype, device=device)
    index = torch.tensor([0, 1, 1, 2, 3, 3, 0, 2], device=device)

    out = pyg_lib.ops.gather_coo(src, index)
    expected = _gather_coo_ref(src, index)
    assert out.size() == (8, 3)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_gather_coo_forward_1d(device):
    src = torch.randn(5, device=device)
    index = torch.tensor([0, 0, 1, 2, 2, 2, 3, 4, 4], device=device)

    out = pyg_lib.ops.gather_coo(src, index)
    expected = _gather_coo_ref(src, index)
    assert out.size() == (9,)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_gather_coo_forward_2d_broadcast_index(device):
    """1-D ``index`` broadcasts over a 2-D ``src``; output trailing dim
    matches ``src``'s trailing dim.
    """
    src = torch.randn(4, 8, device=device)
    index = torch.tensor([0, 1, 1, 2, 3, 3, 0, 2], device=device)

    out = pyg_lib.ops.gather_coo(src, index)
    expected = _gather_coo_ref(src, index)
    assert out.size() == (8, 8)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_gather_coo_forward_k1_small_trailing(device):
    """K==1 path: 1-D src, 1-D index."""
    src = torch.tensor([10.0, 20.0, 30.0, 40.0], device=device)
    index = torch.tensor([0, 1, 1, 2, 3, 3, 0, 2], device=device)

    out = pyg_lib.ops.gather_coo(src, index)
    expected = torch.tensor(
        [10.0, 20.0, 20.0, 30.0, 40.0, 40.0, 10.0, 30.0],
        device=device,
    )
    torch.testing.assert_close(out, expected)


@withCUDA
def test_gather_coo_forward_k_large_broadcast(device):
    """K>1 path: large trailing feature dim exercises broadcast kernel."""
    src = torch.randn(5, 64, device=device)
    index = torch.tensor([0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 4], device=device)

    out = pyg_lib.ops.gather_coo(src, index)
    expected = _gather_coo_ref(src, index)
    assert out.size() == (11, 64)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_gather_coo_forward_inverse_of_segment_sum_signature(device):
    """``segment_sum_coo`` output rows have shape ``[N, *trailing]``; passing
    it through ``gather_coo`` with the same ``index`` recovers a tensor of
    shape ``[E, *trailing]`` (the backward of segment_sum_coo).
    """
    src = torch.randn(8, 3, device=device)
    index = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3], device=device)

    summed = pyg_lib.ops.segment_sum_coo(src, index)  # [4, 3]
    scattered = pyg_lib.ops.gather_coo(summed, index)  # [8, 3]
    assert scattered.size() == (8, 3)
    expected = _gather_coo_ref(summed, index)
    torch.testing.assert_close(scattered, expected)


@withCUDA
def test_gather_coo_empty_index(device):
    """Empty index -> empty output (shape ``[0, *trailing]``)."""
    src = torch.randn(3, 4, device=device)
    index = torch.empty(0, dtype=torch.long, device=device)

    out = pyg_lib.ops.gather_coo(src, index)
    assert out.size() == (0, 4)


# ---------------------------------------------------------------------------
# gather_coo — backward
# ---------------------------------------------------------------------------


@withCUDA
def test_gather_coo_backward_gradcheck(device):
    src = torch.randn(
        4,
        3,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    index = torch.tensor([0, 1, 1, 2, 3, 3, 0, 2], device=device)

    def fn(s):
        return pyg_lib.ops.gather_coo(s, index)

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_gather_coo_backward_gradcheck_1d(device):
    src = torch.randn(
        5,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    index = torch.tensor([0, 0, 1, 2, 2, 2, 3, 4, 4], device=device)

    def fn(s):
        return pyg_lib.ops.gather_coo(s, index)

    assert torch.autograd.gradcheck(fn, (src,))


# ---------------------------------------------------------------------------
# gradgradcheck — symmetric pair (each op's backward calls the other)
# ---------------------------------------------------------------------------


@withCUDA
def test_segment_sum_coo_gradgradcheck(device):
    """``SegmentSumCOO.backward`` invokes ``gather_coo``; gradgradcheck
    exercises ``gather_coo``'s backward (which in turn invokes
    ``segment_sum_coo``).
    """
    src = torch.randn(
        6,
        3,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    index = torch.tensor([0, 0, 1, 1, 1, 2], device=device)

    def fn(s):
        return pyg_lib.ops.segment_sum_coo(s, index)

    assert torch.autograd.gradgradcheck(fn, (src,))


@withCUDA
def test_gather_coo_gradgradcheck(device):
    """``GatherCOO.backward`` invokes ``segment_sum_coo``; gradgradcheck
    exercises ``segment_sum_coo``'s backward (which in turn invokes
    ``gather_coo``).
    """
    src = torch.randn(
        4,
        3,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    index = torch.tensor([0, 1, 1, 2, 3, 3, 0, 2], device=device)

    def fn(s):
        return pyg_lib.ops.gather_coo(s, index)

    assert torch.autograd.gradgradcheck(fn, (src,))


# ---------------------------------------------------------------------------
# segment_mean_coo — forward
# ---------------------------------------------------------------------------


@withCUDA
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_segment_mean_coo_forward_dtypes(dtype, device):
    torch.manual_seed(0)
    src = torch.randn(8, 4, dtype=dtype, device=device)
    index = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3], device=device)

    out = pyg_lib.ops.segment_mean_coo(src, index)
    expected = _segment_mean_coo_ref(src, index)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_mean_coo_forward_1d(device):
    src = torch.randn(8, device=device)
    index = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3], device=device)

    out = pyg_lib.ops.segment_mean_coo(src, index)
    expected = _segment_mean_coo_ref(src, index)
    assert out.size() == (4,)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_mean_coo_forward_2d_broadcast_index(device):
    """1-D ``index`` broadcasts over a 2-D ``src`` along ``dim = 0``."""
    src = torch.randn(6, 4, device=device)
    index = torch.tensor([0, 0, 1, 1, 1, 2], device=device)

    out = pyg_lib.ops.segment_mean_coo(src, index)
    expected = _segment_mean_coo_ref(src, index)
    assert out.size() == (3, 4)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_mean_coo_forward_k1_small_trailing(device):
    """K==1 path: no trailing feature dims (purely 1-D)."""
    src = torch.randn(10, device=device)
    index = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], device=device)

    out = pyg_lib.ops.segment_mean_coo(src, index)
    expected = _segment_mean_coo_ref(src, index)
    assert out.size() == (5,)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_mean_coo_forward_k_large_broadcast(device):
    """K>1 path: large trailing feature dim exercises the broadcast kernel."""
    src = torch.randn(12, 64, device=device)
    index = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4], device=device)

    out = pyg_lib.ops.segment_mean_coo(src, index)
    expected = _segment_mean_coo_ref(src, index)
    assert out.size() == (5, 64)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_mean_coo_dim_size_auto_infer(device):
    src = torch.randn(6, device=device)
    index = torch.tensor([0, 1, 1, 2, 2, 3], device=device)

    out = pyg_lib.ops.segment_mean_coo(src, index)
    # Auto-inferred dim_size = index.max() + 1 = 4.
    assert out.size(-1) == 4
    expected = _segment_mean_coo_ref(src, index)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_mean_coo_dim_size_explicit(device):
    src = torch.randn(6, device=device)
    index = torch.tensor([0, 1, 1, 2, 2, 3], device=device)

    out = pyg_lib.ops.segment_mean_coo(src, index, dim_size=6)
    assert out.size(-1) == 6
    expected = _segment_mean_coo_ref(src, index, dim_size=6)
    torch.testing.assert_close(out, expected)
    # Trailing empty buckets should be exactly zero (count masked to 1).
    torch.testing.assert_close(out[4:], torch.zeros(2, device=device))


@withCUDA
def test_segment_mean_coo_empty_rows_in_middle(device):
    """Row 1 has no entries — its output must be zero (count = 0 -> 0/1)."""
    src = torch.randn(5, 4, device=device)
    index = torch.tensor([0, 0, 2, 2, 2], device=device)

    out = pyg_lib.ops.segment_mean_coo(src, index)
    expected = _segment_mean_coo_ref(src, index)
    assert out.size() == (3, 4)
    torch.testing.assert_close(out, expected)
    # Row 1 is empty -> zeros.
    torch.testing.assert_close(out[1], torch.zeros(4, device=device))


@withCUDA
def test_segment_mean_coo_empty_input(device):
    """Empty source and index — output is all-zero of the requested shape."""
    src = torch.empty(0, 4, device=device)
    index = torch.empty(0, dtype=torch.long, device=device)

    out = pyg_lib.ops.segment_mean_coo(src, index, dim_size=3)
    assert out.size() == (3, 4)
    torch.testing.assert_close(out, torch.zeros(3, 4, device=device))


# ---------------------------------------------------------------------------
# segment_mean_coo — backward
# ---------------------------------------------------------------------------


@withCUDA
def test_segment_mean_coo_backward_gradcheck(device):
    src = torch.randn(
        6,
        3,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    index = torch.tensor([0, 0, 1, 1, 1, 2], device=device)

    def fn(s):
        return pyg_lib.ops.segment_mean_coo(s, index)

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_segment_mean_coo_backward_gradcheck_with_dim_size(device):
    """Gradcheck with explicit (larger) ``dim_size`` -> empty trailing rows."""
    src = torch.randn(
        5,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    index = torch.tensor([0, 0, 2, 2, 2], device=device)

    def fn(s):
        return pyg_lib.ops.segment_mean_coo(s, index, dim_size=5)

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_segment_mean_coo_backward_gradcheck_empty_rows(device):
    """Empty-row fixture under gradcheck."""
    src = torch.randn(
        5,
        4,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    index = torch.tensor([0, 0, 2, 2, 2], device=device)

    def fn(s):
        return pyg_lib.ops.segment_mean_coo(s, index)

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_segment_mean_coo_backward_empty_input(device):
    """Empty ``src`` and ``index`` — exercises the ``numel() > 0`` guard in
    backward. The forward is well-defined (all-zero output); the backward
    must not crash on the empty grad and should return a correctly-shaped
    zero gradient.
    """
    src = torch.zeros(
        0,
        4,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    index = torch.empty(0, dtype=torch.long, device=device)

    out = pyg_lib.ops.segment_mean_coo(src, index, dim_size=3)
    assert out.size() == (3, 4)
    # Sum to a scalar and backprop — exercises backward on an empty grad.
    out.sum().backward()
    assert src.grad is not None
    assert src.grad.size() == src.size()
    torch.testing.assert_close(src.grad, torch.zeros_like(src))
