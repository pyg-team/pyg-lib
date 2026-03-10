import pytest
import torch

import pyg_lib
from pyg_lib.testing import withCUDA


@withCUDA
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_grid_cluster_2d(dtype: torch.dtype, device: torch.device) -> None:
    pos = torch.tensor(
        [[0.0, 0.0], [0.1, 0.1], [0.5, 0.5], [1.0, 1.0], [1.1, 1.1]],
        dtype=dtype, device=device)
    size = torch.tensor([0.5, 0.5], dtype=dtype, device=device)

    out = pyg_lib.ops.grid_cluster(pos, size)

    # Points (0,0) and (0.1,0.1) should be in the same cluster
    assert out[0] == out[1]
    # Point (0.5,0.5) should be in a different cluster from (0,0)
    assert out[0] != out[2]
    # Points (1.0,1.0) and (1.1,1.1) should be in the same cluster
    assert out[3] == out[4]


@withCUDA
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_grid_cluster_3d(dtype: torch.dtype, device: torch.device) -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [1.0, 1.0, 1.0]],
                       dtype=dtype, device=device)
    size = torch.tensor([0.5, 0.5, 0.5], dtype=dtype, device=device)

    out = pyg_lib.ops.grid_cluster(pos, size)

    assert out[0] == out[1]
    assert out[0] != out[2]


@withCUDA
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_grid_cluster_with_start_end(dtype: torch.dtype,
                                     device: torch.device) -> None:
    pos = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=dtype,
                       device=device)
    size = torch.tensor([0.5, 0.5], dtype=dtype, device=device)
    start = torch.tensor([0.0, 0.0], dtype=dtype, device=device)
    end = torch.tensor([1.0, 1.0], dtype=dtype, device=device)

    out = pyg_lib.ops.grid_cluster(pos, size, start, end)

    assert out.shape == (3, )
    assert out.dtype == torch.long


@withCUDA
def test_grid_cluster_cpu_cuda_parity(device: torch.device) -> None:
    pos = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    size = torch.tensor([0.5, 0.5])

    out_cpu = pyg_lib.ops.grid_cluster(pos, size)
    out_dev = pyg_lib.ops.grid_cluster(pos.to(device), size.to(device))

    assert torch.equal(out_cpu, out_dev.cpu())
