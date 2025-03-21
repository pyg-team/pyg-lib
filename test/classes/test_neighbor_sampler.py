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
        'A__to__B': torch.tensor([0, 2, 2, 4]),
        'B__to__A': torch.tensor([0, 1, 3]),
    }
    col = {
        'A__to__B': torch.tensor([1, 0, 1, 0]),
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
    seed_node = {'A': torch.tensor([2, 0]), 'B': torch.tensor([1])}
    seed_time = {'A': torch.tensor([3, 1]), 'B': torch.tensor([2])}
    (row, col, node_id, edge_id, batch, num_sampled_nodes,
     num_sampled_edges) = sampler.sample(num_neighbors, seed_node, seed_time,
                                         True, 'last', True)
    assert torch.equal(row['A__to__B'], torch.tensor([0, 1]))
    assert torch.equal(row['B__to__A'], torch.tensor([0, 1, 2]))
    assert torch.equal(col['A__to__B'], torch.tensor([1, 2]))
    assert torch.equal(col['B__to__A'], torch.tensor([2, 3, 4]))
    assert torch.equal(edge_id['A__to__B'], torch.tensor([3, 0]))
    assert torch.equal(edge_id['B__to__A'], torch.tensor([1, 0, 1]))
    assert torch.equal(node_id['A'], torch.tensor([2, 0, 1, 0, 1]))
    assert torch.equal(node_id['B'], torch.tensor([1, 0, 1]))
    assert torch.equal(batch['A'], torch.tensor([0, 1, 2, 0, 1]))
    assert torch.equal(batch['B'], torch.tensor([2, 0, 1]))
    assert num_sampled_nodes == {'A': [2, 1, 2], 'B': [1, 2, 0]}
    assert num_sampled_edges == {'A__to__B': [0, 2, 0], 'B__to__A': [0, 1, 2]}


def test_hetero_neighbor_sampler_static_sample() -> None:
    node_types = ['A', 'B']
    edge_types = [('A', 'to', 'B'), ('B', 'to', 'A')]
    rowptr = {
        'A__to__B': torch.tensor([0, 2, 2, 4]),
        'B__to__A': torch.tensor([0, 1, 3]),
    }
    col = {
        'A__to__B': torch.tensor([1, 0, 1, 0]),
        'B__to__A': torch.tensor([0, 1, 2]),
    }

    Sampler = torch.classes.pyg.HeteroNeighborSampler
    sampler = Sampler(node_types, edge_types, rowptr, col, None, None, None)

    num_neighbors = {
        'A__to__B': [1, 1],
        'B__to__A': [1, 1],
    }
    seed_node = {'A': torch.tensor([2, 0]), 'B': torch.tensor([1])}
    (row, col, node_id, edge_id, batch, num_sampled_nodes,
     num_sampled_edges) = sampler.sample(num_neighbors, seed_node, None, True,
                                         'uniform', True)
    # Outputs are non-deterministic so we just assert that we get them
    assert row
    assert col
    assert node_id
    assert edge_id
    assert batch
    assert num_sampled_nodes
    assert num_sampled_edges
