import torch

import pyg_lib
from pyg_lib.testing import withCUDA


@withCUDA
def test_knn_graph_basic(device: torch.device) -> None:
    x = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        device=device,
    )
    edge_index = pyg_lib.ops.knn_graph(x, k=2)

    assert edge_index.shape[0] == 2
    # No self-loops by default:
    assert (edge_index[0] != edge_index[1]).all()
    # Each node should have at most k=2 neighbors:
    for i in range(x.size(0)):
        assert (edge_index[0] == i).sum() <= 2


@withCUDA
def test_knn_graph_with_loop(device: torch.device) -> None:
    x = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        device=device,
    )
    edge_index = pyg_lib.ops.knn_graph(x, k=2, loop=True)

    assert edge_index.shape[0] == 2
    # With loop=True, self-loops should be present:
    self_loops = (edge_index[0] == edge_index[1]).sum()
    assert self_loops > 0


@withCUDA
def test_knn_graph_flow(device: torch.device) -> None:
    x = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        device=device,
    )
    e_s2t = pyg_lib.ops.knn_graph(x, k=1, flow='source_to_target')
    e_t2s = pyg_lib.ops.knn_graph(x, k=1, flow='target_to_source')

    # The two should be flipped versions of each other:
    assert torch.equal(e_s2t[0], e_t2s[1])
    assert torch.equal(e_s2t[1], e_t2s[0])


@withCUDA
def test_knn_graph_batched(device: torch.device) -> None:
    x = torch.randn(20, 3, device=device)
    ptr = torch.tensor([0, 10, 20], dtype=torch.long, device=device)

    edge_index = pyg_lib.ops.knn_graph(x, k=3, ptr=ptr)

    assert edge_index.shape[0] == 2
    # Batch 0 queries should only connect to batch 0:
    batch0_mask = edge_index[0] < 10
    assert (edge_index[1, batch0_mask] < 10).all()
    # Batch 1 queries should only connect to batch 1:
    batch1_mask = edge_index[0] >= 10
    assert (edge_index[1, batch1_mask] >= 10).all()
