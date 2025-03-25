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
    assert sum(batch['A'] == 0) == 2
    assert sum(batch['B'] == 0) == 1
    assert sum(batch['A'] == 1) == 2
    assert sum(batch['B'] == 1) == 1
    assert sum(batch['A'] == 2) == 1
    assert sum(batch['B'] == 2) == 1
    assert sum(batch['A'] == 3) == 2
    assert sum(batch['B'] == 3) == 2
    assert sum(batch['A'] == 4) == 1
    assert sum(batch['B'] == 4) == 2
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


def test_metapath_tracker() -> None:
    edge_types = [
        ('A', 'to', 'B'),
        ('B', 'to', 'C'),
    ]
    num_neighbors = {
        'A__to__B': [10, 0],
        'B__to__C': [0, 2],
    }
    seed_node_types = ['A']
    tracker = torch.classes.pyg.MetapathTracker(edge_types, num_neighbors,
                                                seed_node_types)
    b1A = tracker.init_batch(1, 'A', 1)
    b2A = tracker.init_batch(2, 'A', 10)
    assert b1A == b2A
    b1AB = tracker.get_neighbor_metapath(b1A, 'A__to__B')
    assert b1AB != b1A
    assert tracker.get_sample_size(1, b1A, ('A', 'to', 'B')) == 10
    assert tracker.get_sample_size(2, b2A, ('A', 'to', 'B')) == 100
    tracker.report_sample_size(1, b1AB, 5)
    tracker.report_sample_size(2, b1AB, 25)
    b1ABC = tracker.get_neighbor_metapath(b1AB, 'B__to__C')
    assert b1ABC != b1AB
    assert b1ABC != b1A
    assert tracker.get_sample_size(1, b1AB, ('B', 'to', 'C')) == 20
    assert tracker.get_sample_size(2, b1AB, ('B', 'to', 'C')) == 200
