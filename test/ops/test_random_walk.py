import torch

import pyg_lib
from pyg_lib.testing import withCUDA


def _make_cycle_graph(num_nodes, device):
    """Create a cycle graph 0-1-2-...-N-1-0 in CSR format."""
    rows = []
    cols = []
    for i in range(num_nodes):
        rows.extend([i, i])
        cols.extend([(i - 1) % num_nodes, (i + 1) % num_nodes])
    # Sort by row for CSR
    edge_pairs = sorted(zip(rows, cols))
    rows = [p[0] for p in edge_pairs]
    cols = [p[1] for p in edge_pairs]

    col = torch.tensor(cols, dtype=torch.long, device=device)
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    for r in rows:
        rowptr[r + 1] += 1
    rowptr = rowptr.cumsum(0)
    return rowptr, col


@withCUDA
def test_random_walk_basic(device: torch.device) -> None:
    rowptr, col = _make_cycle_graph(4, device)
    seed = torch.arange(4, dtype=torch.long, device=device)

    out = pyg_lib.sampler.random_walk(rowptr, col, seed, walk_length=5)
    assert out.shape == (4, 6)
    # First column should be the seed nodes:
    assert torch.equal(out[:, 0], seed)


@withCUDA
def test_random_walk_return_edge_indices(device: torch.device) -> None:
    rowptr, col = _make_cycle_graph(4, device)
    seed = torch.arange(4, dtype=torch.long, device=device)

    node_seq, edge_seq = pyg_lib.sampler.random_walk(
        rowptr,
        col,
        seed,
        walk_length=5,
        return_edge_indices=True,
    )
    assert node_seq.shape == (4, 6)
    assert edge_seq.shape == (4, 5)
    assert torch.equal(node_seq[:, 0], seed)

    # All edge indices should be non-negative (no isolated nodes):
    assert (edge_seq >= 0).all()

    # Verify edge indices are consistent with the walk:
    for i in range(4):
        for j in range(5):
            eidx = edge_seq[i, j].item()
            target_node = node_seq[i, j + 1].item()
            assert col[eidx].item() == target_node


@withCUDA
def test_random_walk_no_edge_indices_by_default(device: torch.device) -> None:
    rowptr, col = _make_cycle_graph(4, device)
    seed = torch.arange(4, dtype=torch.long, device=device)

    out = pyg_lib.sampler.random_walk(rowptr, col, seed, walk_length=3)
    # Default should return a single Tensor, not a tuple:
    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 4)
