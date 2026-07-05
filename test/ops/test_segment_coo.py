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


# ---------------------------------------------------------------------------
# segment_min_coo — reference
# ---------------------------------------------------------------------------


def _segment_min_coo_ref(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int = None,
):
    """Pure-PyTorch reference for :func:`segment_min_coo`.

    COO ops always reduce along ``dim = index.dim() - 1`` with a sorted
    ``index``. Returns ``(value, argindex)``:
      * ``value[..., j, ...]`` is the minimum of ``src`` entries whose
        broadcast index equals ``j`` along ``dim``.
      * ``argindex[..., j, ...]`` is the position along ``dim`` of the
        *first-match* min entry (CPU upstream contract).
      * Empty buckets get ``value == 0`` and ``argindex == src.size(dim)``
        (upstream sentinel).
    """
    dim = index.dim() - 1
    bcast_index = _broadcast_index(index, src)

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

    # Per-element Python loop over src along the reduction dim, tracking
    # running min + first-match argindex per output position.
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
                value[out_idx_t] = flat_src[k]
                argindex[out_idx_t] = i
            else:
                cur = value[out_idx_t]
                v = flat_src[k]
                if bool(v < cur):
                    value[out_idx_t] = v
                    argindex[out_idx_t] = i

    return value, argindex


# ---------------------------------------------------------------------------
# segment_min_coo — forward
# ---------------------------------------------------------------------------


@withCUDA
@pytest.mark.parametrize(
    'dtype',
    [torch.int32, torch.int64, torch.float32, torch.float64],
)
def test_segment_min_coo_forward_dtypes(dtype, device):
    """Forward correctness on a unique-value fixture across dtypes.

    Unique values mean argindex is unambiguous on both CPU and CUDA.
    """
    if dtype in (torch.int32, torch.int64):
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
        torch.manual_seed(0)
        flat = (torch.randperm(32, device=device) - 16).to(dtype)
        src = flat.view(8, 4)
    # Sorted ascending index — required by segment_*_coo.
    index = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3], device=device)

    value, arg = pyg_lib.ops.segment_min_coo(src, index)
    ref_value, ref_arg = _segment_min_coo_ref(src, index)
    torch.testing.assert_close(value, ref_value)
    # Unique values -> exact argindex equivalence on both CPU and CUDA.
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_segment_min_coo_forward_1d(device):
    """Unique-value 1-D fixture exercising K==1 path."""
    torch.manual_seed(0)
    src = torch.randperm(8, device=device).to(torch.float32) - 4
    index = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3], device=device)

    value, arg = pyg_lib.ops.segment_min_coo(src, index)
    ref_value, ref_arg = _segment_min_coo_ref(src, index)
    assert value.size() == (4,)
    assert arg.size() == (4,)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_segment_min_coo_forward_2d_broadcast_index(device):
    """1-D ``index`` broadcasts over a 2-D ``src`` along ``dim = 0``."""
    torch.manual_seed(0)
    src = (torch.randperm(6 * 4, device=device).to(torch.float32) - 12).view(
        6,
        4,
    )
    index = torch.tensor([0, 0, 1, 1, 1, 2], device=device)

    value, arg = pyg_lib.ops.segment_min_coo(src, index)
    ref_value, ref_arg = _segment_min_coo_ref(src, index)
    assert value.size() == (3, 4)
    assert arg.size() == (3, 4)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_segment_min_coo_forward_k_large_broadcast(device):
    """K>1 path: large trailing feature dim exercises broadcast kernel."""
    torch.manual_seed(0)
    src = (
        torch.randperm(12 * 64, device=device).to(torch.float32) - 384
    ).view(12, 64)
    index = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4], device=device)

    value, arg = pyg_lib.ops.segment_min_coo(src, index)
    ref_value, ref_arg = _segment_min_coo_ref(src, index)
    assert value.size() == (5, 64)
    assert arg.size() == (5, 64)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_segment_min_coo_dim_size_auto_infer(device):
    src = torch.tensor([5.0, -1.0, 3.0, -7.0, 2.0, 9.0], device=device)
    index = torch.tensor([0, 1, 1, 2, 2, 3], device=device)

    value, arg = pyg_lib.ops.segment_min_coo(src, index)
    # Auto-inferred dim_size = index.max() + 1 = 4.
    assert value.size(-1) == 4
    assert arg.size(-1) == 4
    ref_value, ref_arg = _segment_min_coo_ref(src, index)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_segment_min_coo_dim_size_explicit_empty_buckets(device):
    """Explicit ``dim_size`` larger than implicit — trailing buckets are
    empty. Upstream convention: value == 0, argindex == sentinel
    (== src.size(dim)).
    """
    src = torch.tensor([5.0, -1.0, 3.0, -7.0, 2.0, 9.0], device=device)
    index = torch.tensor([0, 1, 1, 2, 2, 3], device=device)
    sentinel = src.size(-1)  # 6

    value, arg = pyg_lib.ops.segment_min_coo(src, index, dim_size=6)
    assert value.size(-1) == 6
    assert arg.size(-1) == 6
    ref_value, ref_arg = _segment_min_coo_ref(src, index, dim_size=6)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)
    # Trailing empty buckets at positions 4, 5 must have value 0 and
    # argindex == sentinel.
    for p in [4, 5]:
        assert value[p].item() == 0
        assert arg[p].item() == sentinel


@withCUDA
def test_segment_min_coo_empty_rows_in_middle(device):
    """Empty rows fixture: row 1 has no entries -> value 0 + sentinel arg."""
    torch.manual_seed(0)
    src = (torch.randperm(5 * 4, device=device).to(torch.float32) - 10).view(
        5,
        4,
    )
    # Row 1 entirely skipped.
    index = torch.tensor([0, 0, 2, 2, 2], device=device)
    sentinel = src.size(0)  # 5

    value, arg = pyg_lib.ops.segment_min_coo(src, index)
    ref_value, ref_arg = _segment_min_coo_ref(src, index)
    assert value.size() == (3, 4)
    assert arg.size() == (3, 4)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)
    # Row 1 has no contributors -> zero value row + sentinel arg row.
    torch.testing.assert_close(value[1], torch.zeros(4, device=device))
    torch.testing.assert_close(
        arg[1],
        torch.full((4,), sentinel, dtype=torch.long, device=device),
    )


@withCUDA
def test_segment_min_coo_empty_input(device):
    """Empty source + empty index -> all-zero value, all-sentinel arg."""
    src = torch.empty(0, 4, device=device)
    index = torch.empty(0, dtype=torch.long, device=device)
    sentinel = src.size(0)  # 0

    value, arg = pyg_lib.ops.segment_min_coo(src, index, dim_size=3)
    assert value.size() == (3, 4)
    assert arg.size() == (3, 4)
    torch.testing.assert_close(value, torch.zeros(3, 4, device=device))
    torch.testing.assert_close(
        arg,
        torch.full((3, 4), sentinel, dtype=torch.long, device=device),
    )


@withCUDA
@withSeed
def test_segment_min_coo_argindex_ties_returns_valid(device):
    """Tied values: validity-only assertion (CUDA atomic ordering is
    non-deterministic).
    """
    # Bucket 0: positions 0, 1, 2 all with value 1.0 (tied min).
    # Bucket 1: positions 3, 4 with value 2.0 (tied min).
    # Bucket 2: empty.
    # Bucket 3: position 5 with unique value 7.0.
    src = torch.tensor(
        [1.0, 1.0, 1.0, 2.0, 2.0, 7.0],
        device=device,
    )
    index = torch.tensor([0, 0, 0, 1, 1, 3], device=device)
    sentinel = src.size(0)  # 6

    value, arg = pyg_lib.ops.segment_min_coo(src, index, dim_size=4)
    # Value must equal the true per-bucket min regardless of tie-break.
    expected_value = torch.tensor([1.0, 2.0, 0.0, 7.0], device=device)
    torch.testing.assert_close(value, expected_value)
    # Bucket 0 arg must be in {0, 1, 2}; bucket 1 in {3, 4}.
    assert int(arg[0].item()) in (0, 1, 2)
    assert int(arg[1].item()) in (3, 4)
    # Bucket 2 is empty -> sentinel.
    assert int(arg[2].item()) == sentinel
    # Bucket 3 has unique value -> position 5.
    assert int(arg[3].item()) == 5
    # Every non-sentinel arg must in fact attain the bucket's min value.
    for j in range(value.size(0)):
        a = int(arg[j].item())
        if a == sentinel:
            continue
        assert src[a].item() == value[j].item(), (
            f'arg[{j}]={a} points to src value {src[a].item()} but bucket '
            f'min is {value[j].item()}'
        )


@withCUDA
def test_segment_min_coo_arg_non_differentiable(device):
    """``arg`` must have ``requires_grad=False`` even when ``value`` does."""
    src = torch.randn(6, dtype=torch.double, device=device, requires_grad=True)
    index = torch.tensor([0, 0, 1, 1, 1, 2], device=device)

    value, arg = pyg_lib.ops.segment_min_coo(src, index)
    assert value.requires_grad
    assert not arg.requires_grad
    assert arg.dtype in (torch.long, torch.int64)


# ---------------------------------------------------------------------------
# segment_min_coo — backward
# ---------------------------------------------------------------------------


@withCUDA
def test_segment_min_coo_backward_gradcheck(device):
    """Gradcheck on the value output. Argindex is excluded via ``[0]``.

    Uses unique-valued src so the active argindex is deterministic and the
    finite-difference numerical Jacobian aligns with the analytical one.
    """
    torch.manual_seed(0)
    src = (
        (torch.randperm(6 * 3, device=device).to(torch.double) - 9)
        .view(6, 3)
        .requires_grad_(True)
    )
    index = torch.tensor([0, 0, 1, 1, 1, 2], device=device)

    def fn(s):
        return pyg_lib.ops.segment_min_coo(s, index)[0]

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_segment_min_coo_backward_gradcheck_with_dim_size(device):
    """Gradcheck with explicit (larger) ``dim_size`` exercising empty
    trailing rows. Their argindex points at the sentinel and the
    ``+1``/``narrow`` backward pattern drops that slot — so the gradient
    w.r.t. ``src`` for empty buckets is zero.
    """
    torch.manual_seed(0)
    src = (
        torch.randperm(6, device=device).to(torch.double) - 3
    ).requires_grad_(True)
    index = torch.tensor([0, 0, 1, 1, 1, 2], device=device)

    def fn(s):
        return pyg_lib.ops.segment_min_coo(s, index, dim_size=6)[0]

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_segment_min_coo_backward_gradcheck_empty_rows(device):
    """Empty-row fixture under gradcheck (row 1 has no entries)."""
    torch.manual_seed(0)
    src = (
        (torch.randperm(5 * 4, device=device).to(torch.double) - 10)
        .view(5, 4)
        .requires_grad_(True)
    )
    index = torch.tensor([0, 0, 2, 2, 2], device=device)

    def fn(s):
        return pyg_lib.ops.segment_min_coo(s, index)[0]

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_segment_min_coo_backward_empty_input(device):
    """Empty ``src`` and ``index`` — exercises the ``numel() > 0`` guard.
    Backward must not crash on empty grad and must return a correctly-shaped
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

    value, arg = pyg_lib.ops.segment_min_coo(src, index, dim_size=3)
    assert value.size() == (3, 4)
    assert arg.size() == (3, 4)
    # Sum to a scalar and backprop — exercises backward on empty grad.
    value.sum().backward()
    assert src.grad is not None
    assert src.grad.size() == src.size()
    torch.testing.assert_close(src.grad, torch.zeros_like(src))


# ---------------------------------------------------------------------------
# segment_max_coo — reference
# ---------------------------------------------------------------------------


def _segment_max_coo_ref(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int = None,
):
    """Pure-PyTorch reference for :func:`segment_max_coo`.

    COO ops always reduce along ``dim = index.dim() - 1`` with a sorted
    ``index``. Returns ``(value, argindex)``:
      * ``value[..., j, ...]`` is the maximum of ``src`` entries whose
        broadcast index equals ``j`` along ``dim``.
      * ``argindex[..., j, ...]`` is the position along ``dim`` of the
        *first-match* max entry (CPU upstream contract).
      * Empty buckets get ``value == 0`` and ``argindex == src.size(dim)``
        (upstream sentinel).
    """
    dim = index.dim() - 1
    bcast_index = _broadcast_index(index, src)

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

    # Per-element Python loop over src along the reduction dim, tracking
    # running max + first-match argindex per output position.
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
                value[out_idx_t] = flat_src[k]
                argindex[out_idx_t] = i
            else:
                cur = value[out_idx_t]
                v = flat_src[k]
                if bool(v > cur):
                    value[out_idx_t] = v
                    argindex[out_idx_t] = i

    return value, argindex


# ---------------------------------------------------------------------------
# segment_max_coo — forward
# ---------------------------------------------------------------------------


@withCUDA
@pytest.mark.parametrize(
    'dtype',
    [torch.int32, torch.int64, torch.float32, torch.float64],
)
def test_segment_max_coo_forward_dtypes(dtype, device):
    """Forward correctness on a unique-value fixture across dtypes.

    Unique values mean argindex is unambiguous on both CPU and CUDA.
    """
    if dtype in (torch.int32, torch.int64):
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
        torch.manual_seed(0)
        flat = (torch.randperm(32, device=device) - 16).to(dtype)
        src = flat.view(8, 4)
    # Sorted ascending index — required by segment_*_coo.
    index = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3], device=device)

    value, arg = pyg_lib.ops.segment_max_coo(src, index)
    ref_value, ref_arg = _segment_max_coo_ref(src, index)
    torch.testing.assert_close(value, ref_value)
    # Unique values -> exact argindex equivalence on both CPU and CUDA.
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_segment_max_coo_forward_1d(device):
    """Unique-value 1-D fixture exercising K==1 path."""
    torch.manual_seed(0)
    src = torch.randperm(8, device=device).to(torch.float32) - 4
    index = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3], device=device)

    value, arg = pyg_lib.ops.segment_max_coo(src, index)
    ref_value, ref_arg = _segment_max_coo_ref(src, index)
    assert value.size() == (4,)
    assert arg.size() == (4,)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_segment_max_coo_forward_2d_broadcast_index(device):
    """1-D ``index`` broadcasts over a 2-D ``src`` along ``dim = 0``."""
    torch.manual_seed(0)
    src = (torch.randperm(6 * 4, device=device).to(torch.float32) - 12).view(
        6,
        4,
    )
    index = torch.tensor([0, 0, 1, 1, 1, 2], device=device)

    value, arg = pyg_lib.ops.segment_max_coo(src, index)
    ref_value, ref_arg = _segment_max_coo_ref(src, index)
    assert value.size() == (3, 4)
    assert arg.size() == (3, 4)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_segment_max_coo_forward_k_large_broadcast(device):
    """K>1 path: large trailing feature dim exercises broadcast kernel."""
    torch.manual_seed(0)
    src = (
        torch.randperm(12 * 64, device=device).to(torch.float32) - 384
    ).view(12, 64)
    index = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4], device=device)

    value, arg = pyg_lib.ops.segment_max_coo(src, index)
    ref_value, ref_arg = _segment_max_coo_ref(src, index)
    assert value.size() == (5, 64)
    assert arg.size() == (5, 64)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_segment_max_coo_dim_size_auto_infer(device):
    src = torch.tensor([5.0, -1.0, 3.0, -7.0, 2.0, 9.0], device=device)
    index = torch.tensor([0, 1, 1, 2, 2, 3], device=device)

    value, arg = pyg_lib.ops.segment_max_coo(src, index)
    # Auto-inferred dim_size = index.max() + 1 = 4.
    assert value.size(-1) == 4
    assert arg.size(-1) == 4
    ref_value, ref_arg = _segment_max_coo_ref(src, index)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_segment_max_coo_dim_size_explicit_empty_buckets(device):
    """Explicit ``dim_size`` larger than implicit — trailing buckets are
    empty. Upstream convention: value == 0, argindex == sentinel
    (== src.size(dim)).
    """
    src = torch.tensor([5.0, -1.0, 3.0, -7.0, 2.0, 9.0], device=device)
    index = torch.tensor([0, 1, 1, 2, 2, 3], device=device)
    sentinel = src.size(-1)  # 6

    value, arg = pyg_lib.ops.segment_max_coo(src, index, dim_size=6)
    assert value.size(-1) == 6
    assert arg.size(-1) == 6
    ref_value, ref_arg = _segment_max_coo_ref(src, index, dim_size=6)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)
    # Trailing empty buckets at positions 4, 5 must have value 0 and
    # argindex == sentinel.
    for p in [4, 5]:
        assert value[p].item() == 0
        assert arg[p].item() == sentinel


@withCUDA
def test_segment_max_coo_empty_rows_in_middle(device):
    """Empty rows fixture: row 1 has no entries -> value 0 + sentinel arg."""
    torch.manual_seed(0)
    src = (torch.randperm(5 * 4, device=device).to(torch.float32) - 10).view(
        5,
        4,
    )
    # Row 1 entirely skipped.
    index = torch.tensor([0, 0, 2, 2, 2], device=device)
    sentinel = src.size(0)  # 5

    value, arg = pyg_lib.ops.segment_max_coo(src, index)
    ref_value, ref_arg = _segment_max_coo_ref(src, index)
    assert value.size() == (3, 4)
    assert arg.size() == (3, 4)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)
    # Row 1 has no contributors -> zero value row + sentinel arg row.
    torch.testing.assert_close(value[1], torch.zeros(4, device=device))
    torch.testing.assert_close(
        arg[1],
        torch.full((4,), sentinel, dtype=torch.long, device=device),
    )


@withCUDA
def test_segment_max_coo_empty_input(device):
    """Empty source + empty index -> all-zero value, all-sentinel arg."""
    src = torch.empty(0, 4, device=device)
    index = torch.empty(0, dtype=torch.long, device=device)
    sentinel = src.size(0)  # 0

    value, arg = pyg_lib.ops.segment_max_coo(src, index, dim_size=3)
    assert value.size() == (3, 4)
    assert arg.size() == (3, 4)
    torch.testing.assert_close(value, torch.zeros(3, 4, device=device))
    torch.testing.assert_close(
        arg,
        torch.full((3, 4), sentinel, dtype=torch.long, device=device),
    )


@withCUDA
@withSeed
def test_segment_max_coo_argindex_ties_returns_valid(device):
    """Tied values: validity-only assertion (CUDA atomic ordering is
    non-deterministic).
    """
    # Bucket 0: positions 0, 1, 2 all with value 1.0 (tied max).
    # Bucket 1: positions 3, 4 with value 2.0 (tied max).
    # Bucket 2: empty.
    # Bucket 3: position 5 with unique value 7.0.
    src = torch.tensor(
        [1.0, 1.0, 1.0, 2.0, 2.0, 7.0],
        device=device,
    )
    index = torch.tensor([0, 0, 0, 1, 1, 3], device=device)
    sentinel = src.size(0)  # 6

    value, arg = pyg_lib.ops.segment_max_coo(src, index, dim_size=4)
    # Value must equal the true per-bucket max regardless of tie-break.
    expected_value = torch.tensor([1.0, 2.0, 0.0, 7.0], device=device)
    torch.testing.assert_close(value, expected_value)
    # Bucket 0 arg must be in {0, 1, 2}; bucket 1 in {3, 4}.
    assert int(arg[0].item()) in (0, 1, 2)
    assert int(arg[1].item()) in (3, 4)
    # Bucket 2 is empty -> sentinel.
    assert int(arg[2].item()) == sentinel
    # Bucket 3 has unique value -> position 5.
    assert int(arg[3].item()) == 5
    # Every non-sentinel arg must in fact attain the bucket's max value.
    for j in range(value.size(0)):
        a = int(arg[j].item())
        if a == sentinel:
            continue
        assert src[a].item() == value[j].item(), (
            f'arg[{j}]={a} points to src value {src[a].item()} but bucket '
            f'max is {value[j].item()}'
        )


@withCUDA
def test_segment_max_coo_arg_non_differentiable(device):
    """``arg`` must have ``requires_grad=False`` even when ``value`` does."""
    src = torch.randn(6, dtype=torch.double, device=device, requires_grad=True)
    index = torch.tensor([0, 0, 1, 1, 1, 2], device=device)

    value, arg = pyg_lib.ops.segment_max_coo(src, index)
    assert value.requires_grad
    assert not arg.requires_grad
    assert arg.dtype in (torch.long, torch.int64)


# ---------------------------------------------------------------------------
# segment_max_coo — backward
# ---------------------------------------------------------------------------


@withCUDA
def test_segment_max_coo_backward_gradcheck(device):
    """Gradcheck on the value output. Argindex is excluded via ``[0]``.

    Uses unique-valued src so the active argindex is deterministic and the
    finite-difference numerical Jacobian aligns with the analytical one.
    """
    torch.manual_seed(0)
    src = (
        (torch.randperm(6 * 3, device=device).to(torch.double) - 9)
        .view(6, 3)
        .requires_grad_(True)
    )
    index = torch.tensor([0, 0, 1, 1, 1, 2], device=device)

    def fn(s):
        return pyg_lib.ops.segment_max_coo(s, index)[0]

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_segment_max_coo_backward_gradcheck_with_dim_size(device):
    """Gradcheck with explicit (larger) ``dim_size`` exercising empty
    trailing rows. Their argindex points at the sentinel and the
    ``+1``/``narrow`` backward pattern drops that slot — so the gradient
    w.r.t. ``src`` for empty buckets is zero.
    """
    torch.manual_seed(0)
    src = (
        torch.randperm(6, device=device).to(torch.double) - 3
    ).requires_grad_(True)
    index = torch.tensor([0, 0, 1, 1, 1, 2], device=device)

    def fn(s):
        return pyg_lib.ops.segment_max_coo(s, index, dim_size=6)[0]

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_segment_max_coo_backward_gradcheck_empty_rows(device):
    """Empty-row fixture under gradcheck (row 1 has no entries)."""
    torch.manual_seed(0)
    src = (
        (torch.randperm(5 * 4, device=device).to(torch.double) - 10)
        .view(5, 4)
        .requires_grad_(True)
    )
    index = torch.tensor([0, 0, 2, 2, 2], device=device)

    def fn(s):
        return pyg_lib.ops.segment_max_coo(s, index)[0]

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_segment_max_coo_backward_empty_input(device):
    """Empty ``src`` and ``index`` — exercises the ``numel() > 0`` guard.
    Backward must not crash on empty grad and must return a correctly-shaped
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

    value, arg = pyg_lib.ops.segment_max_coo(src, index, dim_size=3)
    assert value.size() == (3, 4)
    assert arg.size() == (3, 4)
    # Sum to a scalar and backprop — exercises backward on empty grad.
    value.sum().backward()
    assert src.grad is not None
    assert src.grad.size() == src.size()
    torch.testing.assert_close(src.grad, torch.zeros_like(src))


# ---------------------------------------------------------------------------
# segment_coo dispatcher (commit 14 — Python layer)
# ---------------------------------------------------------------------------


@withCUDA
@pytest.mark.parametrize(
    'reduce',
    ['sum', 'add', 'mean', 'min', 'max'],
)
def test_segment_coo_dispatcher(reduce, device):
    """``segment_coo(src, index, out, dim_size, reduce=...)`` must route to
    the corresponding typed op. For ``min``/``max`` the dispatcher returns
    ``[0]`` (value only), not the ``(value, argindex)`` tuple.

    Note: there is no ``segment_mul_coo``, so ``mul`` is not part of the
    valid reduce set for this dispatcher.
    """
    torch.manual_seed(0)
    # Unique values -> deterministic argindex tie-break across devices.
    src = (torch.randperm(8 * 3, device=device).to(torch.float64) - 12).view(
        8,
        3,
    )
    index = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3], device=device)

    out = pyg_lib.ops.segment_coo(src, index, reduce=reduce)
    if reduce in ('sum', 'add'):
        expected = pyg_lib.ops.segment_sum_coo(src, index)
    elif reduce == 'mean':
        expected = pyg_lib.ops.segment_mean_coo(src, index)
    elif reduce == 'min':
        expected = pyg_lib.ops.segment_min_coo(src, index)[0]
    elif reduce == 'max':
        expected = pyg_lib.ops.segment_max_coo(src, index)[0]
    assert isinstance(out, torch.Tensor)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_coo_dispatcher_unknown_reduce_raises(device):
    """The dispatcher must reject unknown reduce strings with a clear error."""
    src = torch.randn(8, device=device)
    index = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3], device=device)
    with pytest.raises(ValueError):
        pyg_lib.ops.segment_coo(src, index, reduce='unsupported')


@withCUDA
def test_segment_coo_dispatcher_default_reduce_is_sum(device):
    """Default ``reduce`` is ``"sum"`` (upstream convention)."""
    torch.manual_seed(0)
    src = torch.randn(8, 3, device=device)
    index = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3], device=device)

    out = pyg_lib.ops.segment_coo(src, index)
    expected = pyg_lib.ops.segment_sum_coo(src, index)
    torch.testing.assert_close(out, expected)
