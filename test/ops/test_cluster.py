import pytest
import torch

import pyg_lib


@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_grid_cluster_2d(dtype: torch.dtype) -> None:
    pos = torch.tensor(
        [[0.0, 0.0], [0.1, 0.1], [0.5, 0.5], [1.0, 1.0], [1.1, 1.1]],
        dtype=dtype)
    size = torch.tensor([0.5, 0.5], dtype=dtype)

    out = pyg_lib.ops.grid_cluster(pos, size)

    # Points (0,0) and (0.1,0.1) should be in the same cluster
    assert out[0] == out[1]
    # Point (0.5,0.5) should be in a different cluster from (0,0)
    assert out[0] != out[2]
    # Points (1.0,1.0) and (1.1,1.1) should be in the same cluster
    assert out[3] == out[4]


@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_grid_cluster_3d(dtype: torch.dtype) -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [1.0, 1.0, 1.0]],
                       dtype=dtype)
    size = torch.tensor([0.5, 0.5, 0.5], dtype=dtype)

    out = pyg_lib.ops.grid_cluster(pos, size)

    assert out[0] == out[1]
    assert out[0] != out[2]


@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_grid_cluster_with_start_end(dtype: torch.dtype) -> None:
    pos = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=dtype)
    size = torch.tensor([0.5, 0.5], dtype=dtype)
    start = torch.tensor([0.0, 0.0], dtype=dtype)
    end = torch.tensor([1.0, 1.0], dtype=dtype)

    out = pyg_lib.ops.grid_cluster(pos, size, start, end)

    assert out.shape == (3, )
    assert out.dtype == torch.long


def test_grid_cluster_defaults_match_explicit() -> None:
    pos = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    size = torch.tensor([0.5, 0.5])

    out_default = pyg_lib.ops.grid_cluster(pos, size)
    out_explicit = pyg_lib.ops.grid_cluster(pos, size, start=pos.min(0).values,
                                            end=pos.max(0).values)

    assert torch.equal(out_default, out_explicit)
