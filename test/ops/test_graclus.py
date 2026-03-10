import torch

import pyg_lib
from pyg_lib.testing import withCUDA


def _make_graph(device):
    """Create a simple 4-node path graph: 0-1-2-3."""
    col = torch.tensor([1, 0, 2, 1, 3, 2], dtype=torch.long, device=device)
    rowptr = torch.tensor([0, 1, 3, 5, 6], dtype=torch.long, device=device)
    return rowptr, col


@withCUDA
def test_graclus_valid_assignment(device: torch.device) -> None:
    rowptr, col = _make_graph(device)
    out = pyg_lib.ops.graclus_cluster(rowptr, col)

    assert out.shape == (4, )
    # All cluster IDs should be valid node indices
    assert (out >= 0).all()
    assert (out < 4).all()

    # For matched pairs, cluster ID = min(u, v)
    for u in range(4):
        cluster = out[u].item()
        assert cluster <= u or out[cluster].item() == cluster


@withCUDA
def test_graclus_matching_property(device: torch.device) -> None:
    rowptr, col = _make_graph(device)

    # Run multiple times since it's randomized
    for _ in range(10):
        out = pyg_lib.ops.graclus_cluster(rowptr, col)

        # Each node either:
        # 1. Is matched with a neighbor (same cluster ID)
        # 2. Is unmatched (cluster ID == own index)
        for u in range(4):
            cluster = out[u].item()
            if cluster == u:
                continue  # unmatched
            # The matched partner must have the same cluster ID
            assert out[cluster].item() == cluster


@withCUDA
def test_graclus_weighted(device: torch.device) -> None:
    rowptr, col = _make_graph(device)
    # Give edge 1-2 very high weight
    weight = torch.tensor([1.0, 1.0, 100.0, 100.0, 1.0, 1.0], device=device)

    # With high weight on 1-2, they should usually be matched
    matched_count = 0
    for _ in range(20):
        out = pyg_lib.ops.graclus_cluster(rowptr, col, weight=weight)
        if out[1].item() == out[2].item():
            matched_count += 1

    # Should match most of the time
    assert matched_count >= 10


@withCUDA
def test_graclus_disconnected(device: torch.device) -> None:
    # Two disconnected edges: 0-1, 2-3
    rowptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long, device=device)
    col = torch.tensor([1, 0, 3, 2], dtype=torch.long, device=device)

    out = pyg_lib.ops.graclus_cluster(rowptr, col)
    assert out.shape == (4, )
    # Pairs (0,1) and (2,3) should be matched
    assert out[0].item() == out[1].item()
    assert out[2].item() == out[3].item()
