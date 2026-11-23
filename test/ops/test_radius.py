import pytest
import torch

import pyg_lib
from pyg_lib.testing import onlyCUDA, withCUDA


@withCUDA
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_radius_basic(dtype: torch.dtype, device: torch.device) -> None:
    x = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]],
        dtype=dtype,
        device=device,
    )
    y = torch.tensor([[0.5, 0.0]], dtype=dtype, device=device)

    out = pyg_lib.ops.radius(x, y, r=1.5)
    assert out.shape[0] == 2

    # Points at distance 0.5 and 0.5 should be found (x[0] and x[1])
    refs = out[1].sort()[0]
    assert refs.tolist() == [0, 1]
    assert (out[0] == 0).all()


@withCUDA
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_radius_correctness(dtype: torch.dtype, device: torch.device) -> None:
    x = torch.randn(30, 3, dtype=dtype, device=device)
    y = torch.randn(10, 3, dtype=dtype, device=device)
    r = 1.5

    out = pyg_lib.ops.radius(x, y, r=r, max_num_neighbors=100)

    # All returned edges should be within radius
    dists = torch.cdist(y.float(), x.float())
    for idx in range(out.shape[1]):
        qi, ri = out[0, idx].item(), out[1, idx].item()
        assert dists[qi, ri] <= r + 1e-5


@withCUDA
def test_radius_max_num_neighbors(device: torch.device) -> None:
    x = torch.randn(50, 3, device=device)
    y = torch.randn(10, 3, device=device)

    out = pyg_lib.ops.radius(x, y, r=100.0, max_num_neighbors=5)
    # Each query should have at most 5 neighbors
    for i in range(y.size(0)):
        assert (out[0] == i).sum() <= 5


@withCUDA
def test_radius_batched(device: torch.device) -> None:
    x = torch.randn(20, 3, device=device)
    y = torch.randn(15, 3, device=device)
    ptr_x = torch.tensor([0, 10, 20], dtype=torch.long, device=device)
    ptr_y = torch.tensor([0, 8, 15], dtype=torch.long, device=device)

    out = pyg_lib.ops.radius(x, y, r=5.0, ptr_x=ptr_x, ptr_y=ptr_y)

    # Batch 0 queries should only reference batch 0 refs
    batch0_mask = out[0] < 8
    assert (out[1, batch0_mask] < 10).all()

    # Batch 1 queries should only reference batch 1 refs
    batch1_mask = out[0] >= 8
    assert (out[1, batch1_mask] >= 10).all()


@withCUDA
def test_radius_ignore_same_index(device: torch.device) -> None:
    x = torch.randn(10, 3, device=device)

    out = pyg_lib.ops.radius(
        x,
        x,
        r=100.0,
        max_num_neighbors=100,
        ignore_same_index=True,
    )
    # No self-loops
    assert (out[0] != out[1]).all()


@onlyCUDA
@pytest.mark.parametrize('r', [300.0, 2**-14])
def test_radius_half_distance_range(r: float) -> None:
    x = torch.tensor(
        [[0], [7 * r / 8], [r], [2 * r]],
        dtype=torch.half,
        device='cuda',
    )
    out = pyg_lib.ops.radius(x, x[:1], r=r)
    assert out.tolist() == [[0, 0], [0, 1]]


@onlyCUDA
def test_radius_bfloat16_neighbor_limit() -> None:
    # Summing 512 unit squares in bfloat16 stalls at 256.
    x = torch.ones(2, 512, dtype=torch.bfloat16, device='cuda')
    x[1].zero_()
    out = pyg_lib.ops.radius(x, x[1:], r=20, max_num_neighbors=1)
    assert out.tolist() == [[0], [1]]


@onlyCUDA
def test_radius_bfloat16_subtraction() -> None:
    x = torch.tensor([[256], [255]], dtype=torch.bfloat16, device='cuda')
    y = torch.tensor([[-0.5]], dtype=x.dtype, device=x.device)
    out = pyg_lib.ops.radius(x, y, r=256.25)
    assert out.tolist() == [[0], [1]]
