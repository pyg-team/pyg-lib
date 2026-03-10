import pytest
import torch

import pyg_lib
from pyg_lib.testing import withCUDA


@withCUDA
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_nearest_basic(dtype: torch.dtype, device: torch.device) -> None:
    x = torch.tensor([[0.0, 0.0], [3.0, 0.0]], dtype=dtype, device=device)
    y = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=dtype, device=device)

    out = pyg_lib.ops.nearest(x, y)
    assert out.shape == (2, )
    assert out[0].item() == 0  # x[0]=(0,0) nearest to y[0]=(1,0)
    assert out[1].item() == 1  # x[1]=(3,0) nearest to y[1]=(2,0)


@withCUDA
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_nearest_correctness(dtype: torch.dtype, device: torch.device) -> None:
    x = torch.randn(20, 5, dtype=dtype, device=device)
    y = torch.randn(15, 5, dtype=dtype, device=device)

    out = pyg_lib.ops.nearest(x, y)

    # Reference: cdist + argmin
    dists = torch.cdist(x.float(), y.float())
    expected = dists.argmin(dim=1)
    assert torch.equal(out, expected.to(out.device))


@withCUDA
def test_nearest_batched(device: torch.device) -> None:
    x = torch.randn(20, 3, device=device)
    y = torch.randn(15, 3, device=device)
    ptr_x = torch.tensor([0, 10, 20], dtype=torch.long, device=device)
    ptr_y = torch.tensor([0, 8, 15], dtype=torch.long, device=device)

    out = pyg_lib.ops.nearest(x, y, ptr_x=ptr_x, ptr_y=ptr_y)
    assert out.shape == (20, )

    # Batch 0 results should be in [0, 8)
    assert (out[:10] >= 0).all()
    assert (out[:10] < 8).all()

    # Batch 1 results should be in [8, 15)
    assert (out[10:] >= 8).all()
    assert (out[10:] < 15).all()
