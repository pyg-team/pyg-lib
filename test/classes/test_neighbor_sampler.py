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


def test_hetero_neighbor_sampler_temporal_sample() -> None:
    node_types = ['A', 'B']
    edge_types = [('A', 'to', 'B'), ('B', 'to', 'A')]
    rowptr = {
        'A__to__B': torch.tensor([0, 2, 2, 3]),
        'B__to__A': torch.tensor([0, 1, 3]),
    }
    col = {
        'A__to__B': torch.tensor([0, 1, 0]),
        'B__to__A': torch.tensor([0, 1, 2]),
    }
    node_time = {
        'A': torch.tensor([1, 0, 3]),
        'B': torch.tensor([2, 1]),
    }

    Sampler = torch.classes.pyg.HeteroNeighborSampler
    sampler = Sampler(node_types, edge_types, rowptr, col, None, node_time,
                      None)

    num_neighbors = {
        'A__to__B': [1, 2],
        'B__to__A': [2, 1],
    }
    seed_node = {'A': torch.tensor([1, 0])}
    seed_time = {'A': torch.tensor([2, 3])}
    (row, col, node_id, edge_id, batch, num_sampled_nodes,
     num_sampled_edges) = sampler.sample(num_neighbors, seed_node, seed_time,
                                         True, 'last', True)
    print(f'Row {row}')
    print(f'Col {col}')
    print(f'Node_id {node_id}')
    print(f'edge_id {edge_id}')
    print(f'batch {batch}')
    print(f'num_sampled_nodes {num_sampled_nodes}')
    print(f'num_sampled_edges {num_sampled_edges}')
