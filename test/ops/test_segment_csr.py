import pytest
import torch

import pyg_lib
from pyg_lib.testing import withCUDA, withSeed

# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _csr_to_coo(indptr: torch.Tensor) -> torch.Tensor:
    """Converts a 1-D ``indptr`` to a 1-D COO ``index`` of length
    ``indptr[-1]`` via ``torch.repeat_interleave``. For example,
    ``indptr = [0, 2, 2, 5]`` maps to ``[0, 0, 2, 2, 2]``.
    """
    num_rows = indptr.numel() - 1
    arange = torch.arange(num_rows, device=indptr.device)
    return torch.repeat_interleave(arange, indptr.diff())


def _segment_sum_csr_ref(
    src: torch.Tensor,
    indptr: torch.Tensor,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Pure-PyTorch reference for :func:`segment_sum_csr`.

    Reduces along ``dim = indptr.dim() - 1`` with output size along ``dim``
    equal to ``indptr.size(-1) - 1``. The reference converts ``indptr`` (any
    leading-dim broadcast handled by simply taking the last row of strides:
    indptr is required to be either truly 1-D w.r.t. its trailing rows, or
    consistent across leading dims) to a COO index and falls back to
    ``torch.zeros + scatter_add_``.
    """
    # CSR reduces along the trailing dim of ``indptr``.
    dim = indptr.dim() - 1
    num_rows = indptr.size(-1) - 1

    # Compute the output shape: src shape with src.size(dim) -> num_rows.
    out_size = list(src.size())
    out_size[dim] = num_rows

    if out is None:
        out = torch.zeros(out_size, dtype=src.dtype, device=src.device)

    if indptr.dim() == 1:
        # 1-D indptr path: convert to a 1-D COO index and scatter_add along
        # the reduction dim.
        coo_index = _csr_to_coo(indptr)
        # Broadcast coo_index to src's shape if needed.
        if coo_index.dim() < src.dim():
            view = [1] * src.dim()
            view[dim] = -1
            bcast = coo_index.view(view).expand_as(src)
        else:
            bcast = coo_index
        out.scatter_add_(dim, bcast, src)
    else:
        # Multi-dim indptr: iterate the leading dims explicitly.
        # ``indptr`` shares all leading dims with ``src``; the trailing dim
        # describes per-leading-coord row boundaries.
        leading_shape = indptr.shape[:-1]
        for leading_idx in (
            torch.cartesian_prod(
                *[torch.arange(s) for s in leading_shape],
            ).tolist()
            if len(leading_shape) > 0
            else [()]
        ):
            if isinstance(leading_idx, int):
                leading_idx = (leading_idx,)
            row_ptr = indptr[tuple(leading_idx)]
            src_slice = src[tuple(leading_idx)]
            out_slice = out[tuple(leading_idx)]
            coo_index = _csr_to_coo(row_ptr)
            if coo_index.dim() < src_slice.dim():
                view = [1] * src_slice.dim()
                view[0] = -1
                bcast = coo_index.view(view).expand_as(src_slice)
            else:
                bcast = coo_index
            out_slice.scatter_add_(0, bcast, src_slice)
    return out


def _gather_csr_ref(
    src: torch.Tensor,
    indptr: torch.Tensor,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Pure-PyTorch reference for :func:`gather_csr`.

    For each output position ``i`` along ``dim = indptr.dim() - 1``,
    ``out[..., i, ...] = src[..., r, ...]`` where ``r`` is the row containing
    ``i`` (i.e. the unique ``r`` with ``indptr[r] <= i < indptr[r+1]``).

    Computed by repeat-interleaving ``src`` along ``dim`` by row-lengths
    ``indptr.diff()``.
    """
    dim = indptr.dim() - 1
    if indptr.dim() == 1:
        repeats = indptr.diff()
        result = torch.repeat_interleave(src, repeats, dim=dim)
    else:
        # Multi-dim indptr: handle the leading dims explicitly.
        leading_shape = indptr.shape[:-1]
        pieces = []
        for leading_idx in (
            torch.cartesian_prod(
                *[torch.arange(s) for s in leading_shape],
            ).tolist()
            if len(leading_shape) > 0
            else [()]
        ):
            if isinstance(leading_idx, int):
                leading_idx = (leading_idx,)
            row_ptr = indptr[tuple(leading_idx)]
            src_slice = src[tuple(leading_idx)]
            pieces.append(
                torch.repeat_interleave(src_slice, row_ptr.diff(), dim=0),
            )
        # Stack back along the leading dim — assumes one leading dim.
        result = torch.stack(pieces, dim=0).view(
            *leading_shape,
            *pieces[0].shape,
        )
    if out is not None:
        out.copy_(result)
        return out
    return result


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------


def test_segment_add_csr_is_segment_sum_csr_alias():
    assert pyg_lib.ops.segment_add_csr is pyg_lib.ops.segment_sum_csr


# ---------------------------------------------------------------------------
# segment_sum_csr — forward
# ---------------------------------------------------------------------------


@withCUDA
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_segment_sum_csr_forward_dtypes(dtype, device):
    torch.manual_seed(0)
    src = torch.randn(8, 4, dtype=dtype, device=device)
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)

    out = pyg_lib.ops.segment_sum_csr(src, indptr)
    expected = _segment_sum_csr_ref(src, indptr)
    assert out.size() == (4, 4)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_sum_csr_forward_1d(device):
    src = torch.randn(8, device=device)
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)

    out = pyg_lib.ops.segment_sum_csr(src, indptr)
    expected = _segment_sum_csr_ref(src, indptr)
    assert out.size() == (4,)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_sum_csr_forward_k1_small_trailing(device):
    """K==1 path: 1-D src, 1-D indptr — exercises the per-row CUDA kernel."""
    src = torch.randn(10, device=device)
    indptr = torch.tensor([0, 2, 4, 6, 8, 10], device=device)

    out = pyg_lib.ops.segment_sum_csr(src, indptr)
    expected = _segment_sum_csr_ref(src, indptr)
    assert out.size() == (5,)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_sum_csr_forward_k_large_broadcast(device):
    """K>1 path: large trailing feature dim exercises the broadcast kernel."""
    src = torch.randn(12, 64, device=device)
    indptr = torch.tensor([0, 2, 5, 7, 9, 12], device=device)

    out = pyg_lib.ops.segment_sum_csr(src, indptr)
    expected = _segment_sum_csr_ref(src, indptr)
    assert out.size() == (5, 64)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_sum_csr_forward_matches_segment_coo(device):
    """The plan's primary parity test: ``segment_sum_csr`` must agree with
    ``segment_sum_coo`` invoked on the COO-equivalent index built by
    ``repeat_interleave(arange(num_rows), indptr.diff())``.
    """
    torch.manual_seed(0)
    src = torch.randn(8, 4, device=device)
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)
    coo_index = _csr_to_coo(indptr)

    out_csr = pyg_lib.ops.segment_sum_csr(src, indptr)
    out_coo = pyg_lib.ops.segment_sum_coo(src, coo_index)
    torch.testing.assert_close(out_csr, out_coo)


@withCUDA
def test_segment_sum_csr_empty_rows_in_middle(device):
    """Plan-mandated empty-rows fixture: ``indptr = [0, 2, 2, 5]`` —
    row 1 is empty and its output row must be zero.
    """
    src = torch.randn(5, 4, device=device)
    indptr = torch.tensor([0, 2, 2, 5], device=device)

    out = pyg_lib.ops.segment_sum_csr(src, indptr)
    expected = _segment_sum_csr_ref(src, indptr)
    assert out.size() == (3, 4)
    torch.testing.assert_close(out, expected)
    # Row 1 is empty -> zeros.
    torch.testing.assert_close(out[1], torch.zeros(4, device=device))


@withCUDA
def test_segment_sum_csr_all_rows_empty(device):
    """All rows empty -> output is all-zero of the expected shape."""
    src = torch.empty(0, 4, device=device)
    indptr = torch.tensor([0, 0, 0, 0], device=device)

    out = pyg_lib.ops.segment_sum_csr(src, indptr)
    assert out.size() == (3, 4)
    torch.testing.assert_close(out, torch.zeros(3, 4, device=device))


@withCUDA
def test_segment_sum_csr_indptr_broadcast_via_expand(device):
    """Indptr broadcasting: an expanded (non-contiguous) ``indptr`` should
    work — the dispatcher must call ``.contiguous()`` at the kernel boundary.
    """
    torch.manual_seed(0)
    # Two "graphs" sharing the same row layout. The natural way to express
    # this is to expand a 1-D indptr to (B, R+1) and let the kernel reduce
    # over the trailing dim for each batch entry.
    src = torch.randn(2, 8, 4, device=device)
    base_indptr = torch.tensor([0, 3, 5, 6, 8], device=device)
    indptr = base_indptr.unsqueeze(0).expand(2, -1)
    # Sanity: the input is intentionally non-contiguous.
    assert not indptr.is_contiguous()

    out = pyg_lib.ops.segment_sum_csr(src, indptr)
    expected = _segment_sum_csr_ref(src, indptr.contiguous())
    assert out.size() == (2, 4, 4)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_sum_csr_stress_short_and_huge_rows(device):
    """Mix of many short rows and a few huge rows — exercises both fast paths
    in the CUDA kernel (per-row sequential vs broadcast).
    """
    torch.manual_seed(0)
    # 50 single-element rows, 1 small row, 2 huge rows.
    row_lengths = [1] * 50 + [3] + [200, 500]
    indptr_vals = [0]
    for r in row_lengths:
        indptr_vals.append(indptr_vals[-1] + r)
    indptr = torch.tensor(indptr_vals, device=device)
    total = indptr_vals[-1]
    src = torch.randn(total, 8, device=device)

    out = pyg_lib.ops.segment_sum_csr(src, indptr)
    expected = _segment_sum_csr_ref(src, indptr)
    assert out.size() == (len(row_lengths), 8)
    # Loose tolerance: huge rows accumulate in different orders on CUDA vs CPU.
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


@withCUDA
def test_segment_sum_csr_with_out_argument(device):
    """``out=...`` overrides allocation; output should be written in place."""
    src = torch.randn(8, 4, device=device)
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)
    out = torch.zeros(4, 4, device=device)

    result = pyg_lib.ops.segment_sum_csr(src, indptr, out=out)
    expected = _segment_sum_csr_ref(src, indptr)
    torch.testing.assert_close(out, expected)
    # The returned tensor should be the same storage as ``out``.
    assert result.data_ptr() == out.data_ptr()


# ---------------------------------------------------------------------------
# segment_sum_csr — backward
# ---------------------------------------------------------------------------


@withCUDA
def test_segment_sum_csr_backward_gradcheck(device):
    src = torch.randn(
        8,
        3,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)

    def fn(s):
        return pyg_lib.ops.segment_sum_csr(s, indptr)

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_segment_sum_csr_backward_gradcheck_empty_rows(device):
    """Empty-row fixture under gradcheck — row 1 has no entries."""
    src = torch.randn(
        5,
        4,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    indptr = torch.tensor([0, 2, 2, 5], device=device)

    def fn(s):
        return pyg_lib.ops.segment_sum_csr(s, indptr)

    assert torch.autograd.gradcheck(fn, (src,))


# ---------------------------------------------------------------------------
# gather_csr — forward
# ---------------------------------------------------------------------------


@withCUDA
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_gather_csr_forward_dtypes(dtype, device):
    torch.manual_seed(0)
    src = torch.randn(4, 3, dtype=dtype, device=device)
    indptr = torch.tensor([0, 2, 5, 6, 8], device=device)

    out = pyg_lib.ops.gather_csr(src, indptr)
    expected = _gather_csr_ref(src, indptr)
    assert out.size() == (8, 3)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_gather_csr_forward_1d(device):
    src = torch.tensor([10.0, 20.0, 30.0, 40.0], device=device)
    indptr = torch.tensor([0, 2, 5, 6, 8], device=device)

    out = pyg_lib.ops.gather_csr(src, indptr)
    # Expected: row 0 (val 10) repeated 2x, row 1 (val 20) 3x, row 2 (val 30)
    # 1x, row 3 (val 40) 2x.
    expected = torch.tensor(
        [10.0, 10.0, 20.0, 20.0, 20.0, 30.0, 40.0, 40.0],
        device=device,
    )
    torch.testing.assert_close(out, expected)


@withCUDA
def test_gather_csr_forward_k_large_broadcast(device):
    """K>1 path: large trailing feature dim exercises broadcast kernel."""
    src = torch.randn(5, 64, device=device)
    indptr = torch.tensor([0, 2, 3, 7, 10, 12], device=device)

    out = pyg_lib.ops.gather_csr(src, indptr)
    expected = _gather_csr_ref(src, indptr)
    assert out.size() == (12, 64)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_gather_csr_empty_rows(device):
    """Empty-row fixture: ``indptr = [0, 2, 2, 5]`` — row 1 contributes
    nothing to the gathered output.
    """
    src = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        device=device,
    )
    indptr = torch.tensor([0, 2, 2, 5], device=device)

    out = pyg_lib.ops.gather_csr(src, indptr)
    expected = _gather_csr_ref(src, indptr)
    assert out.size() == (5, 2)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_gather_csr_inverse_of_segment_sum_signature(device):
    """``segment_sum_csr`` output rows have shape ``[N, *trailing]``; passing
    it through ``gather_csr`` with the same ``indptr`` recovers a tensor of
    shape ``[E, *trailing]`` (the backward of segment_sum_csr).
    """
    src = torch.randn(8, 3, device=device)
    indptr = torch.tensor([0, 2, 5, 6, 8], device=device)

    summed = pyg_lib.ops.segment_sum_csr(src, indptr)  # [4, 3]
    scattered = pyg_lib.ops.gather_csr(summed, indptr)  # [8, 3]
    assert scattered.size() == (8, 3)
    expected = _gather_csr_ref(summed, indptr)
    torch.testing.assert_close(scattered, expected)


@withCUDA
def test_gather_csr_indptr_broadcast_via_expand(device):
    """Expanded (non-contiguous) ``indptr`` — dispatcher must
    ``.contiguous()`` at the kernel boundary.
    """
    src = torch.randn(2, 4, 3, device=device)
    base_indptr = torch.tensor([0, 2, 5, 6, 8], device=device)
    indptr = base_indptr.unsqueeze(0).expand(2, -1)
    assert not indptr.is_contiguous()

    out = pyg_lib.ops.gather_csr(src, indptr)
    expected = _gather_csr_ref(src, indptr.contiguous())
    assert out.size() == (2, 8, 3)
    torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# gather_csr — backward
# ---------------------------------------------------------------------------


@withCUDA
def test_gather_csr_backward_gradcheck(device):
    src = torch.randn(
        4,
        3,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    indptr = torch.tensor([0, 2, 5, 6, 8], device=device)

    def fn(s):
        return pyg_lib.ops.gather_csr(s, indptr)

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_gather_csr_backward_gradcheck_empty_rows(device):
    """Empty-row fixture under gradcheck — row 1 has no entries, so its
    gradient contribution is zero.
    """
    src = torch.randn(
        3,
        2,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    indptr = torch.tensor([0, 2, 2, 5], device=device)

    def fn(s):
        return pyg_lib.ops.gather_csr(s, indptr)

    assert torch.autograd.gradcheck(fn, (src,))


# ---------------------------------------------------------------------------
# gradgradcheck — symmetric pair (each op's backward calls the other)
# ---------------------------------------------------------------------------


@withCUDA
def test_segment_sum_csr_gradgradcheck(device):
    """``SegmentSumCSR.backward`` invokes ``gather_csr``; gradgradcheck
    exercises ``gather_csr``'s backward (which in turn invokes
    ``segment_sum_csr``).
    """
    src = torch.randn(
        8,
        3,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)

    def fn(s):
        return pyg_lib.ops.segment_sum_csr(s, indptr)

    assert torch.autograd.gradgradcheck(fn, (src,))


@withCUDA
def test_gather_csr_gradgradcheck(device):
    """``GatherCSR.backward`` invokes ``segment_sum_csr``; gradgradcheck
    exercises ``segment_sum_csr``'s backward (which in turn invokes
    ``gather_csr``).
    """
    src = torch.randn(
        4,
        3,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    indptr = torch.tensor([0, 2, 5, 6, 8], device=device)

    def fn(s):
        return pyg_lib.ops.gather_csr(s, indptr)

    assert torch.autograd.gradgradcheck(fn, (src,))


@withCUDA
def test_segment_sum_csr_gradgradcheck_empty_rows(device):
    """Gradgradcheck on the empty-rows fixture — exercises the symmetric
    backward pair in the presence of zero-length rows.
    """
    src = torch.randn(
        5,
        4,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    indptr = torch.tensor([0, 2, 2, 5], device=device)

    def fn(s):
        return pyg_lib.ops.segment_sum_csr(s, indptr)

    assert torch.autograd.gradgradcheck(fn, (src,))


# ---------------------------------------------------------------------------
# segment_mean_csr — reference
# ---------------------------------------------------------------------------


def _segment_mean_csr_ref(
    src: torch.Tensor,
    indptr: torch.Tensor,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Pure-PyTorch reference for :func:`segment_mean_csr`.

    Computes the per-row sum via :func:`_segment_sum_csr_ref` and divides by
    per-row counts (``indptr.diff()``). Counts for empty rows are masked to
    ``1`` so empty rows yield exactly ``0`` (matching upstream semantics).
    """
    dim = indptr.dim() - 1
    summed = _segment_sum_csr_ref(src, indptr)
    # Per-row counts: same leading dims as indptr, trailing length R.
    counts = indptr.diff().to(src.dtype)
    # Broadcast counts along the reduction dim of ``summed``.
    shape = [1] * summed.dim()
    shape[dim] = counts.size(-1)
    # If indptr has leading dims, expand counts to match.
    if counts.dim() > 1:
        # Shape counts to (..., R, 1, 1, ...) — broadcast across trailing dims.
        view = list(counts.shape) + [1] * (summed.dim() - counts.dim())
        counts = counts.view(view)
    else:
        counts = counts.view(shape)
    safe_counts = counts.masked_fill(counts == 0, 1)
    result = summed / safe_counts
    if out is not None:
        out.copy_(result)
        return out
    return result


# ---------------------------------------------------------------------------
# segment_mean_csr — forward
# ---------------------------------------------------------------------------


@withCUDA
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_segment_mean_csr_forward_dtypes(dtype, device):
    torch.manual_seed(0)
    src = torch.randn(8, 4, dtype=dtype, device=device)
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)

    out = pyg_lib.ops.segment_mean_csr(src, indptr)
    expected = _segment_mean_csr_ref(src, indptr)
    assert out.size() == (4, 4)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_mean_csr_forward_1d(device):
    src = torch.randn(8, device=device)
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)

    out = pyg_lib.ops.segment_mean_csr(src, indptr)
    expected = _segment_mean_csr_ref(src, indptr)
    assert out.size() == (4,)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_mean_csr_forward_k1_small_trailing(device):
    """K==1 path: 1-D src, 1-D indptr — exercises the per-row CUDA kernel."""
    src = torch.randn(10, device=device)
    indptr = torch.tensor([0, 2, 4, 6, 8, 10], device=device)

    out = pyg_lib.ops.segment_mean_csr(src, indptr)
    expected = _segment_mean_csr_ref(src, indptr)
    assert out.size() == (5,)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_mean_csr_forward_k_large_broadcast(device):
    """K>1 path: large trailing feature dim exercises the broadcast kernel."""
    src = torch.randn(12, 64, device=device)
    indptr = torch.tensor([0, 2, 5, 7, 9, 12], device=device)

    out = pyg_lib.ops.segment_mean_csr(src, indptr)
    expected = _segment_mean_csr_ref(src, indptr)
    assert out.size() == (5, 64)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_mean_csr_forward_matches_segment_mean_coo(device):
    """Primary parity test: ``segment_mean_csr`` must agree with
    ``segment_mean_coo`` invoked on the COO-equivalent index built by
    ``repeat_interleave(arange(num_rows), indptr.diff())``.
    """
    torch.manual_seed(0)
    src = torch.randn(8, 4, device=device)
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)
    coo_index = _csr_to_coo(indptr)

    out_csr = pyg_lib.ops.segment_mean_csr(src, indptr)
    out_coo = pyg_lib.ops.segment_mean_coo(src, coo_index)
    torch.testing.assert_close(out_csr, out_coo)


@withCUDA
def test_segment_mean_csr_empty_rows_in_middle(device):
    """Plan-mandated empty-rows fixture: ``indptr = [0, 2, 2, 5]`` —
    row 1 is empty and its output row must be zero.
    """
    src = torch.randn(5, 4, device=device)
    indptr = torch.tensor([0, 2, 2, 5], device=device)

    out = pyg_lib.ops.segment_mean_csr(src, indptr)
    expected = _segment_mean_csr_ref(src, indptr)
    assert out.size() == (3, 4)
    torch.testing.assert_close(out, expected)
    # Row 1 is empty -> zeros (count = 0 -> 0 / 1).
    torch.testing.assert_close(out[1], torch.zeros(4, device=device))


@withCUDA
def test_segment_mean_csr_all_rows_empty(device):
    """All rows empty -> output is all-zero of the expected shape."""
    src = torch.empty(0, 4, device=device)
    indptr = torch.tensor([0, 0, 0, 0], device=device)

    out = pyg_lib.ops.segment_mean_csr(src, indptr)
    assert out.size() == (3, 4)
    torch.testing.assert_close(out, torch.zeros(3, 4, device=device))


@withCUDA
def test_segment_mean_csr_forward_matches_segment_mean_coo_empty_rows(device):
    """COO parity on the empty-row fixture."""
    src = torch.randn(5, 4, device=device)
    indptr = torch.tensor([0, 2, 2, 5], device=device)
    coo_index = _csr_to_coo(indptr)

    out_csr = pyg_lib.ops.segment_mean_csr(src, indptr)
    out_coo = pyg_lib.ops.segment_mean_coo(src, coo_index, dim_size=3)
    torch.testing.assert_close(out_csr, out_coo)


# ---------------------------------------------------------------------------
# segment_mean_csr — backward
# ---------------------------------------------------------------------------


@withCUDA
def test_segment_mean_csr_backward_gradcheck(device):
    src = torch.randn(
        8,
        3,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)

    def fn(s):
        return pyg_lib.ops.segment_mean_csr(s, indptr)

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_segment_mean_csr_backward_gradcheck_empty_rows(device):
    """Empty-row fixture under gradcheck — row 1 has no entries."""
    src = torch.randn(
        5,
        4,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    indptr = torch.tensor([0, 2, 2, 5], device=device)

    def fn(s):
        return pyg_lib.ops.segment_mean_csr(s, indptr)

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_segment_mean_csr_backward_empty_input(device):
    """Empty ``src`` and all-zero ``indptr`` — exercises the
    ``grad_in.numel() > 0`` guard in backward. Forward yields an all-zero
    output and backward must not crash on the empty grad, returning a
    correctly-shaped zero gradient.
    """
    src = torch.zeros(
        0,
        4,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    indptr = torch.tensor([0, 0, 0, 0], device=device)

    out = pyg_lib.ops.segment_mean_csr(src, indptr)
    assert out.size() == (3, 4)
    # Sum to a scalar and backprop — exercises backward on an empty grad.
    out.sum().backward()
    assert src.grad is not None
    assert src.grad.size() == src.size()
    torch.testing.assert_close(src.grad, torch.zeros_like(src))


# ---------------------------------------------------------------------------
# segment_min_csr — reference
# ---------------------------------------------------------------------------


def _segment_min_csr_ref(
    src: torch.Tensor,
    indptr: torch.Tensor,
):
    """Pure-PyTorch reference for :func:`segment_min_csr`.

    Reduces along ``dim = indptr.dim() - 1`` with output size along ``dim``
    equal to ``indptr.size(-1) - 1``. Returns ``(value, argindex)``:
      * ``value[..., r, ...]`` is the minimum of ``src`` entries whose
        position along ``dim`` lies in ``[indptr[r], indptr[r+1])``.
      * ``argindex[..., r, ...]`` is the position along ``dim`` of the
        first-match min entry (CPU upstream contract).
      * Empty rows (``indptr[r+1] == indptr[r]``) get ``value == 0`` and
        ``argindex == src.size(dim)`` (upstream sentinel).

    This reference handles the 1-D ``indptr`` case (the variant tested
    below — matching the plan's spec for commit 12).
    """
    assert indptr.dim() == 1, '1-D indptr reference only'
    dim = indptr.dim() - 1  # == 0
    num_rows = indptr.size(-1) - 1
    sentinel = src.size(dim)

    out_size = list(src.size())
    out_size[dim] = num_rows
    value = torch.zeros(out_size, dtype=src.dtype, device=src.device)
    argindex = torch.full(
        out_size,
        sentinel,
        dtype=torch.long,
        device=src.device,
    )

    indptr_cpu = indptr.detach().cpu().tolist()
    for r in range(num_rows):
        lo, hi = indptr_cpu[r], indptr_cpu[r + 1]
        if lo == hi:
            # Empty row -> leave zero value + sentinel arg.
            continue
        seg = src.narrow(dim, lo, hi - lo)
        # min along the reduction dim; ``torch.min`` returns (values, indices)
        # where indices index *into the narrowed segment*, so offset by ``lo``.
        seg_val, seg_arg = seg.min(dim=dim)
        # Assign into the r-th slice of value/argindex along ``dim``.
        value.select(dim, r).copy_(seg_val)
        argindex.select(dim, r).copy_(seg_arg + lo)

    return value, argindex


# ---------------------------------------------------------------------------
# segment_min_csr — forward
# ---------------------------------------------------------------------------


@withCUDA
@pytest.mark.parametrize(
    'dtype',
    [torch.int32, torch.int64, torch.float32, torch.float64],
)
def test_segment_min_csr_forward_dtypes(dtype, device):
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
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)

    value, arg = pyg_lib.ops.segment_min_csr(src, indptr)
    ref_value, ref_arg = _segment_min_csr_ref(src, indptr)
    assert value.size() == (4, 4)
    assert arg.size() == (4, 4)
    torch.testing.assert_close(value, ref_value)
    # Unique values -> exact argindex equivalence on both CPU and CUDA.
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_segment_min_csr_forward_1d(device):
    """Unique-value 1-D fixture exercising K==1 path."""
    torch.manual_seed(0)
    src = torch.randperm(8, device=device).to(torch.float32) - 4
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)

    value, arg = pyg_lib.ops.segment_min_csr(src, indptr)
    ref_value, ref_arg = _segment_min_csr_ref(src, indptr)
    assert value.size() == (4,)
    assert arg.size() == (4,)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_segment_min_csr_forward_k_large_broadcast(device):
    """K>1 path: large trailing feature dim exercises broadcast kernel."""
    torch.manual_seed(0)
    src = (
        torch.randperm(12 * 64, device=device).to(torch.float32) - 384
    ).view(12, 64)
    indptr = torch.tensor([0, 2, 5, 7, 9, 12], device=device)

    value, arg = pyg_lib.ops.segment_min_csr(src, indptr)
    ref_value, ref_arg = _segment_min_csr_ref(src, indptr)
    assert value.size() == (5, 64)
    assert arg.size() == (5, 64)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_segment_min_csr_forward_matches_segment_min_coo(device):
    """Primary parity test: ``segment_min_csr`` must agree with
    ``segment_min_coo`` invoked on the COO-equivalent index built by
    ``repeat_interleave(arange(num_rows), indptr.diff())``.

    Uses a unique-value source so argindex tie-breaks are deterministic on
    both CPU and CUDA.
    """
    torch.manual_seed(0)
    src = (torch.randperm(8 * 4, device=device).to(torch.float32) - 16).view(
        8,
        4,
    )
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)
    coo_index = _csr_to_coo(indptr)

    val_csr, arg_csr = pyg_lib.ops.segment_min_csr(src, indptr)
    val_coo, arg_coo = pyg_lib.ops.segment_min_coo(src, coo_index)
    torch.testing.assert_close(val_csr, val_coo)
    torch.testing.assert_close(arg_csr, arg_coo)


@withCUDA
def test_segment_min_csr_empty_rows_in_middle(device):
    """Plan-mandated empty-rows fixture: ``indptr = [0, 2, 2, 5]`` —
    row 1 is empty; its output value row is zero and arg row is sentinel.
    """
    torch.manual_seed(0)
    src = (torch.randperm(5 * 4, device=device).to(torch.float32) - 10).view(
        5,
        4,
    )
    indptr = torch.tensor([0, 2, 2, 5], device=device)
    sentinel = src.size(0)  # 5

    value, arg = pyg_lib.ops.segment_min_csr(src, indptr)
    ref_value, ref_arg = _segment_min_csr_ref(src, indptr)
    assert value.size() == (3, 4)
    assert arg.size() == (3, 4)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)
    # Row 1 is empty -> zero value row + sentinel arg row.
    torch.testing.assert_close(value[1], torch.zeros(4, device=device))
    torch.testing.assert_close(
        arg[1],
        torch.full((4,), sentinel, dtype=torch.long, device=device),
    )


@withCUDA
@withSeed
def test_segment_min_csr_argindex_ties_returns_valid(device):
    """Tied values: validity-only assertion (CUDA atomic ordering is
    non-deterministic).
    """
    # Row 0: positions 0, 1, 2 all with value 1.0 (tied min).
    # Row 1: positions 3, 4 with value 2.0 (tied min).
    # Row 2: empty.
    # Row 3: position 5 with unique value 7.0.
    src = torch.tensor(
        [1.0, 1.0, 1.0, 2.0, 2.0, 7.0],
        device=device,
    )
    indptr = torch.tensor([0, 3, 5, 5, 6], device=device)
    sentinel = src.size(0)  # 6

    value, arg = pyg_lib.ops.segment_min_csr(src, indptr)
    # Value must equal the true per-row min regardless of tie-break.
    expected_value = torch.tensor([1.0, 2.0, 0.0, 7.0], device=device)
    torch.testing.assert_close(value, expected_value)
    # Row 0 arg must be in {0, 1, 2}; row 1 in {3, 4}.
    assert int(arg[0].item()) in (0, 1, 2)
    assert int(arg[1].item()) in (3, 4)
    # Row 2 is empty -> sentinel.
    assert int(arg[2].item()) == sentinel
    # Row 3 has unique value -> position 5.
    assert int(arg[3].item()) == 5
    # Every non-sentinel arg must attain the row's min value.
    for r in range(value.size(0)):
        a = int(arg[r].item())
        if a == sentinel:
            continue
        assert src[a].item() == value[r].item(), (
            f'arg[{r}]={a} points to src value {src[a].item()} but row '
            f'min is {value[r].item()}'
        )


@withCUDA
def test_segment_min_csr_arg_non_differentiable(device):
    """``arg`` must have ``requires_grad=False`` even when ``value`` does."""
    src = torch.randn(8, dtype=torch.double, device=device, requires_grad=True)
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)

    value, arg = pyg_lib.ops.segment_min_csr(src, indptr)
    assert value.requires_grad
    assert not arg.requires_grad
    assert arg.dtype in (torch.long, torch.int64)


# ---------------------------------------------------------------------------
# segment_min_csr — backward
# ---------------------------------------------------------------------------


@withCUDA
def test_segment_min_csr_backward_gradcheck(device):
    """Gradcheck on the value output. Argindex is excluded via ``[0]``.

    Uses a unique-valued src so the active argindex is deterministic and the
    finite-difference numerical Jacobian aligns with the analytical one.
    """
    torch.manual_seed(0)
    src = (
        (torch.randperm(8 * 3, device=device).to(torch.double) - 12)
        .view(8, 3)
        .requires_grad_(True)
    )
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)

    def fn(s):
        return pyg_lib.ops.segment_min_csr(s, indptr)[0]

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_segment_min_csr_backward_gradcheck_empty_rows(device):
    """Empty-row fixture under gradcheck — row 1 has no entries; its
    argindex points at the sentinel and the ``+1``/``narrow`` backward
    pattern drops that slot.
    """
    torch.manual_seed(0)
    src = (
        (torch.randperm(5 * 4, device=device).to(torch.double) - 10)
        .view(5, 4)
        .requires_grad_(True)
    )
    indptr = torch.tensor([0, 2, 2, 5], device=device)

    def fn(s):
        return pyg_lib.ops.segment_min_csr(s, indptr)[0]

    assert torch.autograd.gradcheck(fn, (src,))


# ---------------------------------------------------------------------------
# segment_max_csr — reference
# ---------------------------------------------------------------------------


def _segment_max_csr_ref(
    src: torch.Tensor,
    indptr: torch.Tensor,
):
    """Pure-PyTorch reference for :func:`segment_max_csr`.

    Reduces along ``dim = indptr.dim() - 1`` with output size along ``dim``
    equal to ``indptr.size(-1) - 1``. Returns ``(value, argindex)``:
      * ``value[..., r, ...]`` is the maximum of ``src`` entries whose
        position along ``dim`` lies in ``[indptr[r], indptr[r+1])``.
      * ``argindex[..., r, ...]`` is the position along ``dim`` of the
        first-match max entry (CPU upstream contract).
      * Empty rows (``indptr[r+1] == indptr[r]``) get ``value == 0`` and
        ``argindex == src.size(dim)`` (upstream sentinel).

    This reference handles the 1-D ``indptr`` case (the variant tested
    below — matching the plan's spec for commit 13).
    """
    assert indptr.dim() == 1, '1-D indptr reference only'
    dim = indptr.dim() - 1  # == 0
    num_rows = indptr.size(-1) - 1
    sentinel = src.size(dim)

    out_size = list(src.size())
    out_size[dim] = num_rows
    value = torch.zeros(out_size, dtype=src.dtype, device=src.device)
    argindex = torch.full(
        out_size,
        sentinel,
        dtype=torch.long,
        device=src.device,
    )

    indptr_cpu = indptr.detach().cpu().tolist()
    for r in range(num_rows):
        lo, hi = indptr_cpu[r], indptr_cpu[r + 1]
        if lo == hi:
            # Empty row -> leave zero value + sentinel arg.
            continue
        seg = src.narrow(dim, lo, hi - lo)
        # max along the reduction dim; ``torch.max`` returns (values, indices)
        # where indices index *into the narrowed segment*, so offset by ``lo``.
        seg_val, seg_arg = seg.max(dim=dim)
        # Assign into the r-th slice of value/argindex along ``dim``.
        value.select(dim, r).copy_(seg_val)
        argindex.select(dim, r).copy_(seg_arg + lo)

    return value, argindex


# ---------------------------------------------------------------------------
# segment_max_csr — forward
# ---------------------------------------------------------------------------


@withCUDA
@pytest.mark.parametrize(
    'dtype',
    [torch.int32, torch.int64, torch.float32, torch.float64],
)
def test_segment_max_csr_forward_dtypes(dtype, device):
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
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)

    value, arg = pyg_lib.ops.segment_max_csr(src, indptr)
    ref_value, ref_arg = _segment_max_csr_ref(src, indptr)
    assert value.size() == (4, 4)
    assert arg.size() == (4, 4)
    torch.testing.assert_close(value, ref_value)
    # Unique values -> exact argindex equivalence on both CPU and CUDA.
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_segment_max_csr_forward_1d(device):
    """Unique-value 1-D fixture exercising K==1 path."""
    torch.manual_seed(0)
    src = torch.randperm(8, device=device).to(torch.float32) - 4
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)

    value, arg = pyg_lib.ops.segment_max_csr(src, indptr)
    ref_value, ref_arg = _segment_max_csr_ref(src, indptr)
    assert value.size() == (4,)
    assert arg.size() == (4,)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_segment_max_csr_forward_k_large_broadcast(device):
    """K>1 path: large trailing feature dim exercises broadcast kernel."""
    torch.manual_seed(0)
    src = (
        torch.randperm(12 * 64, device=device).to(torch.float32) - 384
    ).view(12, 64)
    indptr = torch.tensor([0, 2, 5, 7, 9, 12], device=device)

    value, arg = pyg_lib.ops.segment_max_csr(src, indptr)
    ref_value, ref_arg = _segment_max_csr_ref(src, indptr)
    assert value.size() == (5, 64)
    assert arg.size() == (5, 64)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)


@withCUDA
def test_segment_max_csr_forward_matches_segment_max_coo(device):
    """Primary parity test: ``segment_max_csr`` must agree with
    ``segment_max_coo`` invoked on the COO-equivalent index built by
    ``repeat_interleave(arange(num_rows), indptr.diff())``.

    Uses a unique-value source so argindex tie-breaks are deterministic on
    both CPU and CUDA.
    """
    torch.manual_seed(0)
    src = (torch.randperm(8 * 4, device=device).to(torch.float32) - 16).view(
        8,
        4,
    )
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)
    coo_index = _csr_to_coo(indptr)

    val_csr, arg_csr = pyg_lib.ops.segment_max_csr(src, indptr)
    val_coo, arg_coo = pyg_lib.ops.segment_max_coo(src, coo_index)
    torch.testing.assert_close(val_csr, val_coo)
    torch.testing.assert_close(arg_csr, arg_coo)


@withCUDA
def test_segment_max_csr_empty_rows_in_middle(device):
    """Plan-mandated empty-rows fixture: ``indptr = [0, 2, 2, 5]`` —
    row 1 is empty; its output value row is zero and arg row is sentinel.
    """
    torch.manual_seed(0)
    src = (torch.randperm(5 * 4, device=device).to(torch.float32) - 10).view(
        5,
        4,
    )
    indptr = torch.tensor([0, 2, 2, 5], device=device)
    sentinel = src.size(0)  # 5

    value, arg = pyg_lib.ops.segment_max_csr(src, indptr)
    ref_value, ref_arg = _segment_max_csr_ref(src, indptr)
    assert value.size() == (3, 4)
    assert arg.size() == (3, 4)
    torch.testing.assert_close(value, ref_value)
    torch.testing.assert_close(arg, ref_arg)
    # Row 1 is empty -> zero value row + sentinel arg row.
    torch.testing.assert_close(value[1], torch.zeros(4, device=device))
    torch.testing.assert_close(
        arg[1],
        torch.full((4,), sentinel, dtype=torch.long, device=device),
    )


@withCUDA
@withSeed
def test_segment_max_csr_argindex_ties_returns_valid(device):
    """Tied values: validity-only assertion (CUDA atomic ordering is
    non-deterministic).
    """
    # Row 0: positions 0, 1, 2 all with value 1.0 (tied max).
    # Row 1: positions 3, 4 with value 2.0 (tied max).
    # Row 2: empty.
    # Row 3: position 5 with unique value 7.0.
    src = torch.tensor(
        [1.0, 1.0, 1.0, 2.0, 2.0, 7.0],
        device=device,
    )
    indptr = torch.tensor([0, 3, 5, 5, 6], device=device)
    sentinel = src.size(0)  # 6

    value, arg = pyg_lib.ops.segment_max_csr(src, indptr)
    # Value must equal the true per-row max regardless of tie-break.
    expected_value = torch.tensor([1.0, 2.0, 0.0, 7.0], device=device)
    torch.testing.assert_close(value, expected_value)
    # Row 0 arg must be in {0, 1, 2}; row 1 in {3, 4}.
    assert int(arg[0].item()) in (0, 1, 2)
    assert int(arg[1].item()) in (3, 4)
    # Row 2 is empty -> sentinel.
    assert int(arg[2].item()) == sentinel
    # Row 3 has unique value -> position 5.
    assert int(arg[3].item()) == 5
    # Every non-sentinel arg must attain the row's max value.
    for r in range(value.size(0)):
        a = int(arg[r].item())
        if a == sentinel:
            continue
        assert src[a].item() == value[r].item(), (
            f'arg[{r}]={a} points to src value {src[a].item()} but row '
            f'max is {value[r].item()}'
        )


@withCUDA
def test_segment_max_csr_arg_non_differentiable(device):
    """``arg`` must have ``requires_grad=False`` even when ``value`` does."""
    src = torch.randn(8, dtype=torch.double, device=device, requires_grad=True)
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)

    value, arg = pyg_lib.ops.segment_max_csr(src, indptr)
    assert value.requires_grad
    assert not arg.requires_grad
    assert arg.dtype in (torch.long, torch.int64)


# ---------------------------------------------------------------------------
# segment_max_csr — backward
# ---------------------------------------------------------------------------


@withCUDA
def test_segment_max_csr_backward_gradcheck(device):
    """Gradcheck on the value output. Argindex is excluded via ``[0]``.

    Uses a unique-valued src so the active argindex is deterministic and the
    finite-difference numerical Jacobian aligns with the analytical one.
    """
    torch.manual_seed(0)
    src = (
        (torch.randperm(8 * 3, device=device).to(torch.double) - 12)
        .view(8, 3)
        .requires_grad_(True)
    )
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)

    def fn(s):
        return pyg_lib.ops.segment_max_csr(s, indptr)[0]

    assert torch.autograd.gradcheck(fn, (src,))


@withCUDA
def test_segment_max_csr_backward_gradcheck_empty_rows(device):
    """Empty-row fixture under gradcheck — row 1 has no entries; its
    argindex points at the sentinel and the ``+1``/``narrow`` backward
    pattern drops that slot.
    """
    torch.manual_seed(0)
    src = (
        (torch.randperm(5 * 4, device=device).to(torch.double) - 10)
        .view(5, 4)
        .requires_grad_(True)
    )
    indptr = torch.tensor([0, 2, 2, 5], device=device)

    def fn(s):
        return pyg_lib.ops.segment_max_csr(s, indptr)[0]

    assert torch.autograd.gradcheck(fn, (src,))


# ---------------------------------------------------------------------------
# segment_csr dispatcher (commit 14 — Python layer)
# ---------------------------------------------------------------------------


@withCUDA
@pytest.mark.parametrize(
    'reduce',
    ['sum', 'add', 'mean', 'min', 'max'],
)
def test_segment_csr_dispatcher(reduce, device):
    """``segment_csr(src, indptr, out, reduce=...)`` must route to the
    corresponding typed op. For ``min``/``max`` the dispatcher returns
    ``[0]`` (value only), not the ``(value, argindex)`` tuple.

    Note: there is no ``segment_mul_csr``, so ``mul`` is not part of the
    valid reduce set for this dispatcher.
    """
    torch.manual_seed(0)
    # Unique values -> deterministic argindex tie-break across devices.
    src = (torch.randperm(8 * 3, device=device).to(torch.float64) - 12).view(
        8,
        3,
    )
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)

    out = pyg_lib.ops.segment_csr(src, indptr, reduce=reduce)
    if reduce in ('sum', 'add'):
        expected = pyg_lib.ops.segment_sum_csr(src, indptr)
    elif reduce == 'mean':
        expected = pyg_lib.ops.segment_mean_csr(src, indptr)
    elif reduce == 'min':
        expected = pyg_lib.ops.segment_min_csr(src, indptr)[0]
    elif reduce == 'max':
        expected = pyg_lib.ops.segment_max_csr(src, indptr)[0]
    assert isinstance(out, torch.Tensor)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_segment_csr_dispatcher_unknown_reduce_raises(device):
    """The dispatcher must reject unknown reduce strings with a clear error."""
    src = torch.randn(8, device=device)
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)
    with pytest.raises(ValueError):
        pyg_lib.ops.segment_csr(src, indptr, reduce='unsupported')


@withCUDA
def test_segment_csr_dispatcher_default_reduce_is_sum(device):
    """Default ``reduce`` is ``"sum"`` (upstream convention)."""
    torch.manual_seed(0)
    src = torch.randn(8, 3, device=device)
    indptr = torch.tensor([0, 3, 5, 6, 8], device=device)

    out = pyg_lib.ops.segment_csr(src, indptr)
    expected = pyg_lib.ops.segment_sum_csr(src, indptr)
    torch.testing.assert_close(out, expected)
