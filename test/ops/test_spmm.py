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


def _spmm_mean_ref(
    rowptr: torch.Tensor,
    col: torch.Tensor,
    value: torch.Tensor,
    mat: torch.Tensor,
) -> torch.Tensor:
    out = _spmm_sum_ref(rowptr, col, value, mat)
    count = rowptr.diff().clamp_min(1).to(dtype=mat.dtype)
    return out / _edge_view(count, out)


def _spmm_minmax_ref(
    rowptr: torch.Tensor,
    col: torch.Tensor,
    value: torch.Tensor,
    mat: torch.Tensor,
    reduce: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_size = list(mat.size())
    out_size[-2] = rowptr.numel() - 1
    if reduce == 'min':
        out = torch.full(out_size, float('inf'), dtype=mat.dtype)
    else:
        out = torch.full(out_size, -float('inf'), dtype=mat.dtype)
    arg = torch.full(out_size, col.numel(), dtype=torch.long)

    for row in range(rowptr.numel() - 1):
        for edge in range(int(rowptr[row]), int(rowptr[row + 1])):
            candidate = mat[..., col[edge], :]
            if value is not None:
                candidate = candidate * value[edge]
            current = out[..., row, :]
            better = (
                candidate < current if reduce == 'min' else candidate > current
            )
            out[..., row, :] = torch.where(better, candidate, current)
            arg[..., row, :] = torch.where(
                better,
                torch.full_like(arg[..., row, :], edge),
                arg[..., row, :],
            )

    out = out.masked_fill(arg == col.numel(), 0)
    return out, arg


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


@withSeed
def test_spmm_mean_forward_with_value_and_empty_rows():
    rowptr = torch.tensor([0, 2, 2, 5])
    col = torch.tensor([0, 2, 1, 0, 1])
    value = torch.randn(col.numel())
    mat = torch.randn(2, 3, 4)

    out = pyg_lib.ops.spmm_mean(rowptr, col, value, mat)
    expected = _spmm_mean_ref(rowptr, col, value, mat)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(
        pyg_lib.ops.spmm(rowptr, col, value, mat, reduce='mean'),
        expected,
    )
    torch.testing.assert_close(out[:, 1], torch.zeros_like(out[:, 1]))


@withSeed
def test_spmm_mean_forward_without_value():
    rowptr = torch.tensor([0, 2, 3, 5])
    col = torch.tensor([0, 2, 1, 0, 1])
    mat = torch.randn(3, 4)

    out = pyg_lib.ops.spmm_mean(rowptr, col, None, mat)
    expected = _spmm_mean_ref(rowptr, col, None, mat)
    torch.testing.assert_close(out, expected)


@withSeed
def test_spmm_mean_backward_value_and_mat_gradcheck():
    rowptr = torch.tensor([0, 2, 3, 5])
    col = torch.tensor([0, 2, 1, 0, 1])
    value = torch.randn(col.numel(), dtype=torch.double, requires_grad=True)
    mat = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)

    def func(v, x):
        return pyg_lib.ops.spmm_mean(rowptr, col, v, x)

    assert torch.autograd.gradcheck(func, (value, mat))


@pytest.mark.parametrize('reduce', ['min', 'max'])
@withSeed
def test_spmm_minmax_forward_with_value_arg_and_empty_rows(reduce):
    rowptr = torch.tensor([0, 2, 2, 5])
    col = torch.tensor([0, 2, 1, 0, 1])
    value = torch.tensor([2.0, -1.0, 3.0, 0.5, -2.0])
    mat = torch.tensor(
        [
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
            [[6.0, 3.0], [5.0, 2.0], [4.0, 1.0]],
        ],
    )

    op = pyg_lib.ops.spmm_min if reduce == 'min' else pyg_lib.ops.spmm_max
    out, arg = op(rowptr, col, value, mat)
    expected, expected_arg = _spmm_minmax_ref(rowptr, col, value, mat, reduce)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(arg, expected_arg)
    torch.testing.assert_close(
        pyg_lib.ops.spmm(rowptr, col, value, mat, reduce=reduce),
        expected,
    )
    torch.testing.assert_close(out[:, 1], torch.zeros_like(out[:, 1]))
    torch.testing.assert_close(
        arg[:, 1],
        torch.full_like(arg[:, 1], col.numel()),
    )


@pytest.mark.parametrize('reduce', ['min', 'max'])
@withSeed
def test_spmm_minmax_forward_without_value(reduce):
    rowptr = torch.tensor([0, 2, 3, 5])
    col = torch.tensor([0, 2, 1, 0, 1])
    mat = torch.randn(3, 4)

    op = pyg_lib.ops.spmm_min if reduce == 'min' else pyg_lib.ops.spmm_max
    out, arg = op(rowptr, col, None, mat)
    expected, expected_arg = _spmm_minmax_ref(rowptr, col, None, mat, reduce)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(arg, expected_arg)


@pytest.mark.parametrize('reduce', ['min', 'max'])
@withSeed
def test_spmm_minmax_backward_value_and_mat_gradcheck(reduce):
    rowptr = torch.tensor([0, 2, 3, 5])
    col = torch.tensor([0, 2, 1, 0, 1])
    value = torch.tensor(
        [0.5, -1.5, 2.0, 1.25, -0.75],
        dtype=torch.double,
        requires_grad=True,
    )
    mat = torch.tensor(
        [
            [[1.0, 4.0], [2.0, -5.0], [3.0, 6.0]],
            [[-6.0, 3.0], [5.0, 2.0], [4.0, -2.0]],
        ],
        dtype=torch.double,
        requires_grad=True,
    )

    def func(v, x):
        return pyg_lib.ops.spmm(rowptr, col, v, x, reduce=reduce)

    assert torch.autograd.gradcheck(func, (value, mat))


def test_spmm_unknown_reduce_raises():
    rowptr = torch.tensor([0, 0])
    col = torch.empty(0, dtype=torch.long)
    mat = torch.randn(1, 2)
    with pytest.raises(ValueError, match='Unknown reduce'):
        pyg_lib.ops.spmm(rowptr, col, None, mat, reduce='mul')
