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


def test_fps_ratio_one() -> None:
    # ratio=1.0 should return all points.
    N = 15
    src = torch.randn(N, 3)
    ptr = torch.tensor([0, N], dtype=torch.long)

    out = pyg_lib.ops.fps(src, ptr, ratio=1.0, random_start=False)
    assert out.shape == (N, )
    assert set(out.tolist()) == set(range(N))


def test_fps_single_point_batch() -> None:
    # Edge case: batch with a single point.
    src = torch.randn(1, 3)
    ptr = torch.tensor([0, 1], dtype=torch.long)

    out = pyg_lib.ops.fps(src, ptr, ratio=1.0, random_start=False)
    assert out.shape == (1, )
    assert out[0] == 0


@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_fps_greedy_property(dtype: torch.dtype) -> None:
    # Verify the greedy FPS invariant: each selected point (after the first)
    # must be the farthest from the already-selected set at the time of its
    # selection.
    src = torch.randn(30, 3, dtype=dtype)
    ptr = torch.tensor([0, 30], dtype=torch.long)

    out = pyg_lib.ops.fps(src, ptr, ratio=0.5, random_start=False)

    selected = [out[0].item()]
    for i in range(1, out.shape[0]):
        # Minimum distance from each candidate to the selected set so far:
        sel = src[selected]
        dists = torch.cdist(src.unsqueeze(0), sel.unsqueeze(0)).squeeze(0)
        min_dists = dists.min(dim=1).values
        # The point FPS picked should have the maximum min-distance:
        expected = min_dists.argmax().item()
        assert out[i].item() == expected
        selected.append(out[i].item())
