import pytest
import torch

import pyg_lib
from pyg_lib.testing import withCUDA


@withCUDA
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_knn_basic(dtype: torch.dtype, device: torch.device) -> None:
    x = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        dtype=dtype,
        device=device,
    )
    y = torch.tensor([[0.5, 0.0], [2.5, 0.0]], dtype=dtype, device=device)

    out = pyg_lib.ops.knn(x, y, k=2)
    assert out.shape[0] == 2
    assert out.shape[1] == 4  # 2 query points * k=2

    # Check output format: row 0 = query indices, row 1 = ref indices
    assert (out[0] >= 0).all()
    assert (out[0] < y.size(0)).all()
    assert (out[1] >= 0).all()
    assert (out[1] < x.size(0)).all()


@withCUDA
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_knn_correctness(dtype: torch.dtype, device: torch.device) -> None:
    x = torch.randn(20, 3, dtype=dtype, device=device)
    y = torch.randn(10, 3, dtype=dtype, device=device)
    k = 3

    out = pyg_lib.ops.knn(x, y, k=k)

    # Reference: cdist + topk
    dists = torch.cdist(y.float(), x.float())
    _, ref_idx = dists.topk(k, dim=1, largest=False)

    # Check each query point found the same neighbors
    for i in range(y.size(0)):
        mask = out[0] == i
        found = out[1, mask].sort()[0]
        expected = ref_idx[i].sort()[0].to(found.device)
        assert torch.equal(found, expected)


@withCUDA
def test_knn_batched(device: torch.device) -> None:
    x = torch.randn(20, 3, device=device)
    y = torch.randn(15, 3, device=device)
    ptr_x = torch.tensor([0, 10, 20], dtype=torch.long, device=device)
    ptr_y = torch.tensor([0, 8, 15], dtype=torch.long, device=device)

    out = pyg_lib.ops.knn(x, y, k=2, ptr_x=ptr_x, ptr_y=ptr_y)
    assert out.shape[0] == 2
    assert out.shape[1] == 15 * 2

    # Batch 0 queries should only reference batch 0 refs
    batch0_mask = out[0] < 8
    assert (out[1, batch0_mask] < 10).all()

    # Batch 1 queries should only reference batch 1 refs
    batch1_mask = out[0] >= 8
    assert (out[1, batch1_mask] >= 10).all()
