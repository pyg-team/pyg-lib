import torch

import pyg_lib
from pyg_lib.testing import withCUDA


@withCUDA
def test_edge_sample_count(device: torch.device) -> None:
    if device.type == 'cuda':
        return  # CPU only
    # Graph: node 0 has 3 edges, node 1 has 2 edges
    rowptr = torch.tensor([0, 3, 5], dtype=torch.long, device=device)
    start = torch.tensor([0, 1], dtype=torch.long, device=device)

    out = pyg_lib.ops.edge_sample(start, rowptr, count=2)
    assert out.numel() == 4  # 2 per node * 2 nodes

    # All sampled edge indices should be valid
    assert (out >= 0).all()
    assert (out < 5).all()

    # Node 0 edges in [0, 3), node 1 edges in [3, 5)
    node0_edges = out[:2]
    node1_edges = out[2:]
    assert (node0_edges < 3).all()
    assert (node1_edges >= 3).all()


@withCUDA
def test_edge_sample_factor(device: torch.device) -> None:
    if device.type == 'cuda':
        return  # CPU only
    # Node with 10 edges
    rowptr = torch.tensor([0, 10], dtype=torch.long, device=device)
    start = torch.tensor([0], dtype=torch.long, device=device)

    out = pyg_lib.ops.edge_sample(start, rowptr, count=0, factor=0.5)
    # ceil(0.5 * 10) = 5
    assert out.numel() == 5


@withCUDA
def test_edge_sample_cap(device: torch.device) -> None:
    if device.type == 'cuda':
        return  # CPU only
    # Node with 3 edges, request 10
    rowptr = torch.tensor([0, 3], dtype=torch.long, device=device)
    start = torch.tensor([0], dtype=torch.long, device=device)

    out = pyg_lib.ops.edge_sample(start, rowptr, count=10)
    # Capped at degree = 3
    assert out.numel() == 3
