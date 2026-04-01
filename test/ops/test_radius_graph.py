import torch

import pyg_lib
from pyg_lib.testing import withCUDA


@withCUDA
def test_radius_graph_basic(device: torch.device) -> None:
    x = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]],
        device=device,
    )
    edge_index = pyg_lib.ops.radius_graph(x, r=1.5)

    assert edge_index.shape[0] == 2
    # No self-loops by default:
    assert (edge_index[0] != edge_index[1]).all()

    # Only close points should be connected:
    for idx in range(edge_index.size(1)):
        i, j = edge_index[0, idx].item(), edge_index[1, idx].item()
        dist = (x[i] - x[j]).norm().item()
        assert dist <= 1.5 + 1e-5


@withCUDA
def test_radius_graph_with_loop(device: torch.device) -> None:
    x = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        device=device,
    )
    edge_index = pyg_lib.ops.radius_graph(x, r=1.5, loop=True)

    assert edge_index.shape[0] == 2
    # With loop=True, self-loops should be present:
    self_loops = (edge_index[0] == edge_index[1]).sum()
    assert self_loops > 0


@withCUDA
def test_radius_graph_flow(device: torch.device) -> None:
    x = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        device=device,
    )
    e_s2t = pyg_lib.ops.radius_graph(x, r=1.5, flow='source_to_target')
    e_t2s = pyg_lib.ops.radius_graph(x, r=1.5, flow='target_to_source')

    # The two should be flipped versions of each other:
    # Sort edges for comparison since order may differ
    def sort_edges(e):
        idx = e[0] * 1000 + e[1]
        perm = idx.argsort()
        return e[:, perm]

    e_s2t_sorted = sort_edges(e_s2t)
    e_t2s_flipped = sort_edges(e_t2s.flip(0))
    assert torch.equal(e_s2t_sorted, e_t2s_flipped)


@withCUDA
def test_radius_graph_batched(device: torch.device) -> None:
    x = torch.randn(20, 3, device=device)
    ptr = torch.tensor([0, 10, 20], dtype=torch.long, device=device)

    edge_index = pyg_lib.ops.radius_graph(
        x,
        r=5.0,
        ptr=ptr,
        max_num_neighbors=100,
    )

    assert edge_index.shape[0] == 2
    # Batch 0 queries should only connect to batch 0:
    batch0_mask = edge_index[0] < 10
    assert (edge_index[1, batch0_mask] < 10).all()
    # Batch 1 queries should only connect to batch 1:
    batch1_mask = edge_index[0] >= 10
    assert (edge_index[1, batch1_mask] >= 10).all()


@withCUDA
def test_radius_graph_max_num_neighbors(device: torch.device) -> None:
    x = torch.randn(50, 3, device=device)
    edge_index = pyg_lib.ops.radius_graph(
        x,
        r=100.0,
        max_num_neighbors=5,
    )

    # Each node should have at most 5 neighbors:
    for i in range(x.size(0)):
        assert (edge_index[0] == i).sum() <= 5
