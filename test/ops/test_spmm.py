import pytest
import torch

import pyg_lib
from pyg_lib.testing import withSeed


def _edge_view(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    shape = [1] * target.dim()
    shape[-2] = -1
    return value.view(shape)


def _edge_index_view(
    index: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    shape = [1] * target.dim()
    shape[-2] = -1
    return index.view(shape).expand_as(target)


def _row_from_rowptr(rowptr: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
    row = torch.arange(
        rowptr.numel() - 1,
        dtype=col.dtype,
        device=col.device,
    )
    return torch.repeat_interleave(row, rowptr.diff())


def _spmm_sum_ref(
    rowptr: torch.Tensor,
    col: torch.Tensor,
    value: torch.Tensor,
    mat: torch.Tensor,
) -> torch.Tensor:
    row = _row_from_rowptr(rowptr, col)
    out_size = list(mat.size())
    out_size[-2] = rowptr.numel() - 1
    out = mat.new_zeros(out_size)

    gathered = mat.index_select(-2, col)
    if value is not None:
        gathered = gathered * _edge_view(value, gathered)

    return out.scatter_add_(-2, _edge_index_view(row, gathered), gathered)


def test_spmm_add_is_spmm_sum_alias():
    assert pyg_lib.ops.spmm_add is pyg_lib.ops.spmm_sum


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@withSeed
def test_spmm_sum_forward_with_value(dtype):
    rowptr = torch.tensor([0, 2, 3, 5])
    col = torch.tensor([0, 2, 1, 0, 1])
    value = torch.tensor([1, 2, 4, 1, 3], dtype=dtype)
    mat = torch.tensor([[1, 4], [2, 5], [3, 6]], dtype=dtype)

    out = pyg_lib.ops.spmm_sum(rowptr, col, value, mat)
    expected = _spmm_sum_ref(rowptr, col, value, mat)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(
        pyg_lib.ops.spmm(rowptr, col, value, mat, reduce='add'),
        expected,
    )


@withSeed
def test_spmm_sum_forward_without_value():
    rowptr = torch.tensor([0, 2, 3, 5])
    col = torch.tensor([0, 2, 1, 0, 1])
    mat = torch.randn(3, 4)

    out = pyg_lib.ops.spmm_sum(rowptr, col, None, mat)
    expected = _spmm_sum_ref(rowptr, col, None, mat)
    torch.testing.assert_close(out, expected)


@withSeed
def test_spmm_sum_forward_batched_mat():
    rowptr = torch.tensor([0, 2, 2, 5])
    col = torch.tensor([0, 2, 1, 0, 1])
    value = torch.randn(col.numel())
    mat = torch.randn(2, 3, 4)

    out = pyg_lib.ops.spmm(rowptr, col, value, mat)
    expected = _spmm_sum_ref(rowptr, col, value, mat)
    assert out.size() == (2, 3, 4)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(out[:, 1], torch.zeros_like(out[:, 1]))


@withSeed
def test_spmm_sum_all_rows_empty():
    rowptr = torch.tensor([0, 0, 0])
    col = torch.empty(0, dtype=torch.long)
    value = torch.empty(0)
    mat = torch.randn(4, 3)

    out = pyg_lib.ops.spmm_sum(rowptr, col, value, mat)
    assert out.size() == (2, 3)
    torch.testing.assert_close(out, torch.zeros_like(out))


@withSeed
def test_spmm_sum_backward_value_and_mat_gradcheck():
    rowptr = torch.tensor([0, 2, 3, 5])
    col = torch.tensor([0, 2, 1, 0, 1])
    value = torch.randn(col.numel(), dtype=torch.double, requires_grad=True)
    mat = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)

    def func(v, x):
        return pyg_lib.ops.spmm_sum(rowptr, col, v, x)

    assert torch.autograd.gradcheck(func, (value, mat))


@withSeed
def test_spmm_sum_backward_mat_gradcheck_without_value():
    rowptr = torch.tensor([0, 2, 3, 5])
    col = torch.tensor([0, 2, 1, 0, 1])
    mat = torch.randn(3, 4, dtype=torch.double, requires_grad=True)

    def func(x):
        return pyg_lib.ops.spmm_sum(rowptr, col, None, x)

    assert torch.autograd.gradcheck(func, (mat,))
