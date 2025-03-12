import torch

import pyg_lib  # noqa


def test_neighbor_sampler() -> None:
    rowptr = torch.tensor([0, 2, 4, 6])
    col = torch.tensor([1, 2, 0, 2, 1, 0])

    Sampler = torch.classes.pyg.NeighborSampler
    sampler = Sampler(rowptr, col, None, None, None)
    assert sampler is not None


def test_hetero_neighbor_sampler() -> None:
    node_types = ['A', 'B']
    edge_types = [('A', 'to', 'B'), ('B', 'to', 'A')]
    rowptr = {
        'A__to__B': torch.tensor([0, 1]),
        'B__to__A': torch.tensor([0, 1]),
    }
    col = {
        'A__to__B': torch.tensor([0]),
        'B__to__A': torch.tensor([0]),
    }

    Sampler = torch.classes.pyg.HeteroNeighborSampler
    sampler = Sampler(node_types, edge_types, rowptr, col, None, None, None)
    assert sampler is not None
