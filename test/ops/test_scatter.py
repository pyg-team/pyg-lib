import torch

from pyg_lib.ops import scatter_add, scatter_mean


def test_scatter_add():
    # 1D case
    src = torch.tensor([1, 3, 2, 4, 5, 6])
    index = torch.tensor([0, 1, 0, 1, 1, 3])
    result = scatter_add(src, index)
    expected = torch.tensor([3, 12, 0, 6])
    assert torch.equal(result, expected)

    # 2D case
    src = torch.tensor([[1, 2], [5, 6], [3, 4], [7, 8], [9, 10], [11, 12]])
    index = torch.tensor([0, 1, 0, 1, 1, 3])
    result = scatter_add(src, index, dim=0)
    expected = torch.tensor([[4, 6], [21, 24], [0, 0], [11, 12]])
    assert torch.equal(result, expected)

    # With pre-allocated output
    src = torch.tensor([1, 3, 2, 4, 5, 6])
    index = torch.tensor([0, 1, 0, 1, 1, 3])
    out = torch.zeros(4, dtype=src.dtype)
    result = scatter_add(src, index, out=out)
    expected = torch.tensor([3, 12, 0, 6])
    assert torch.equal(result, expected)
    assert result is out

    # With explicit dim_size
    src = torch.tensor([1, 3, 2])
    index = torch.tensor([0, 2, 1])
    result = scatter_add(src, index, dim_size=5)
    expected = torch.tensor([1, 2, 3, 0, 0])
    assert torch.equal(result, expected)


def test_scatter_mean():
    # 1D case
    src = torch.tensor([1.0, 3.0, 2.0, 4.0, 5.0, 6.0])
    index = torch.tensor([0, 1, 0, 1, 1, 3])
    result = scatter_mean(src, index)
    expected = torch.tensor([1.5, 4.0, 0.0, 6.0])
    assert torch.allclose(result, expected)

    # 2D case
    src = torch.tensor([[1.0, 2.0], [5.0, 6.0], [3.0, 4.0], 
                        [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    index = torch.tensor([0, 1, 0, 1, 1, 3])
    result = scatter_mean(src, index, dim=0)
    expected = torch.tensor([[2.0, 3.0], [7.0, 8.0], [0.0, 0.0], [11.0, 12.0]])
    assert torch.allclose(result, expected)


def test_scatter_edge_cases():
    # Empty tensors
    src = torch.tensor([], dtype=torch.float32)
    index = torch.tensor([], dtype=torch.long)
    result = scatter_add(src, index, dim_size=3)
    expected = torch.zeros(3, dtype=torch.float32)
    assert torch.equal(result, expected)

    # Single element
    src = torch.tensor([5.0])
    index = torch.tensor([2])
    result = scatter_add(src, index, dim_size=4)
    expected = torch.tensor([0.0, 0.0, 5.0, 0.0])
    assert torch.equal(result, expected)


def test_scatter_different_dtypes():
    # Integer tensors
    src = torch.tensor([1, 3, 2, 4], dtype=torch.int32)
    index = torch.tensor([0, 1, 0, 1])
    result = scatter_add(src, index)
    expected = torch.tensor([3, 7], dtype=torch.int32)
    assert torch.equal(result, expected)

    # Float tensors
    src = torch.tensor([1.0, 3.0, 2.0, 4.0], dtype=torch.float64)
    index = torch.tensor([0, 1, 0, 1])
    result = scatter_add(src, index)
    expected = torch.tensor([3.0, 7.0], dtype=torch.float64)
    assert torch.equal(result, expected)


def test_scatter_broadcasting():
    # Test broadcasting of index tensor
    src = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    index = torch.tensor([0, 1, 0])
    result = scatter_add(src, index, dim=0)
    expected = torch.tensor([[8, 10, 12], [4, 5, 6]])
    assert torch.equal(result, expected)


if __name__ == '__main__':
    test_scatter_add()
    test_scatter_mean()
    test_scatter_edge_cases()
    test_scatter_different_dtypes()
    test_scatter_broadcasting()
    print("All scatter tests passed!")