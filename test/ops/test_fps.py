import pytest
import torch

import pyg_lib
from pyg_lib.testing import withCUDA


@withCUDA
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_fps_output_size(dtype: torch.dtype, device: torch.device) -> None:
    N, D = 20, 3
    src = torch.randn(N, D, dtype=dtype, device=device)
    ptr = torch.tensor([0, N], dtype=torch.long, device=device)

    out = pyg_lib.ops.fps(src, ptr, ratio=0.5, random_start=False)
    assert out.shape == (10, )
    assert out.dtype == torch.long
    assert out.min() >= 0
    assert out.max() < N


@withCUDA
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_fps_farthest_property(dtype: torch.dtype,
                               device: torch.device) -> None:
    src = torch.randn(50, 3, dtype=dtype, device=device)
    ptr = torch.tensor([0, 50], dtype=torch.long, device=device)

    out = pyg_lib.ops.fps(src, ptr, ratio=0.2, random_start=False)
    selected = src[out]
    dists = torch.cdist(selected, selected)
    dists.fill_diagonal_(float('inf'))
    min_dist = dists.min()
    assert min_dist > 0


@withCUDA
def test_fps_multi_batch(device: torch.device) -> None:
    src = torch.randn(30, 3, device=device)
    ptr = torch.tensor([0, 10, 30], dtype=torch.long, device=device)

    out = pyg_lib.ops.fps(src, ptr, ratio=0.5, random_start=False)
    # Batch 0: ceil(10 * 0.5) = 5, Batch 1: ceil(20 * 0.5) = 10
    assert out.shape == (15, )
    assert (out[:5] < 10).all()
    assert (out[:5] >= 0).all()
    assert (out[5:] >= 10).all()
    assert (out[5:] < 30).all()


@withCUDA
def test_fps_random_start(device: torch.device) -> None:
    src = torch.randn(20, 3, device=device)
    ptr = torch.tensor([0, 20], dtype=torch.long, device=device)

    out_det = pyg_lib.ops.fps(src, ptr, ratio=0.5, random_start=False)
    assert out_det[0] == 0
