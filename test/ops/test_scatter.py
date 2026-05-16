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
