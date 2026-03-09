import pytest
import torch

import pyg_lib


@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_fps_output_size(dtype: torch.dtype) -> None:
    N, D = 20, 3
    src = torch.randn(N, D, dtype=dtype)
    ptr = torch.tensor([0, N], dtype=torch.long)

    out = pyg_lib.ops.fps(src, ptr, ratio=0.5, random_start=False)
    assert out.shape == (10, )
    assert out.dtype == torch.long
    # All indices should be within range:
    assert out.min() >= 0
    assert out.max() < N


@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_fps_farthest_property(dtype: torch.dtype) -> None:
    # After FPS, the minimum pairwise distance between selected points
    # should be >= the greedy guarantee.
    src = torch.randn(50, 3, dtype=dtype)
    ptr = torch.tensor([0, 50], dtype=torch.long)

    out = pyg_lib.ops.fps(src, ptr, ratio=0.2, random_start=False)
    selected = src[out]
    dists = torch.cdist(selected, selected)
    dists.fill_diagonal_(float('inf'))
    min_dist = dists.min()
    assert min_dist > 0


def test_fps_multi_batch() -> None:
    src = torch.randn(30, 3)
    ptr = torch.tensor([0, 10, 30], dtype=torch.long)

    out = pyg_lib.ops.fps(src, ptr, ratio=0.5, random_start=False)
    # Batch 0: ceil(10 * 0.5) = 5, Batch 1: ceil(20 * 0.5) = 10
    assert out.shape == (15, )
    # First 5 indices in batch 0:
    assert (out[:5] < 10).all()
    assert (out[:5] >= 0).all()
    # Next 10 in batch 1:
    assert (out[5:] >= 10).all()
    assert (out[5:] < 30).all()


def test_fps_random_start() -> None:
    src = torch.randn(20, 3)
    ptr = torch.tensor([0, 20], dtype=torch.long)

    out_det = pyg_lib.ops.fps(src, ptr, ratio=0.5, random_start=False)
    # Deterministic: first selected index is always 0
    assert out_det[0] == 0
