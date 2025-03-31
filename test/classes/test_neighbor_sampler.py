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
        'A__to__B': [1, 1],
        'B__to__A': [2, 1],
    }
    seed_node = {'A': torch.tensor([2, 0]), 'B': torch.tensor([1, 1, 0])}
    seed_time = {'A': torch.tensor([3, 1]), 'B': torch.tensor([2, 4, 3])}
    (row, col, node_id, edge_id, batch, num_sampled_nodes,
     num_sampled_edges) = sampler.sample(num_neighbors, seed_node, seed_time,
                                         True, 'last', True)
    # Due to random shuffle, the output isn't entirely deterministic
    assert row['A__to__B'].shape[0] == sum(num_sampled_edges['A__to__B'])
    assert row['B__to__A'].shape[0] == sum(num_sampled_edges['B__to__A'])
    assert col['A__to__B'].shape[0] == sum(num_sampled_edges['A__to__B'])
    assert col['B__to__A'].shape[0] == sum(num_sampled_edges['B__to__A'])
    assert node_id['A'].shape[0] == sum(num_sampled_nodes['A'])
    assert node_id['B'].shape[0] == sum(num_sampled_nodes['B'])
    # We check that the number of nodes in each batch match the expected count
    # Their internal IDs might be subject to randomness due to the shuffle
    # For batch 0, we sample A2 -> B0 -> A1
    assert (batch['A'] == 0).sum() == 2
    assert (batch['B'] == 0).sum() == 1
    # For batch 1, we sample A0 -> B1 -> A1
    assert (batch['A'] == 1).sum() == 2
    assert (batch['B'] == 1).sum() == 1
    # For batch 2, we sample B1 -> A1
    assert (batch['A'] == 2).sum() == 1
    assert (batch['B'] == 2).sum() == 1
    # For batch 3, we sample B1 -> (A1, A2) -> B0
    assert (batch['A'] == 3).sum() == 2
    assert (batch['B'] == 3).sum() == 2
    # For batch 4, we sample B0 -> A0 -> (B0, B1)
    assert (batch['A'] == 4).sum() == 1
    assert (batch['B'] == 4).sum() == 2
    assert num_sampled_nodes == {'A': [2, 4, 2], 'B': [3, 2, 2]}
    assert (num_sampled_edges == {
        'A__to__B': [0, 2, 4],
        'B__to__A': [0, 4, 2]
    } or num_sampled_edges == {
        'A__to__B': [0, 2, 3],
        'B__to__A': [0, 4, 2]
    })


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
