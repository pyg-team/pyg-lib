import time

import torch

import pyg_lib
from pyg_lib.testing import to_edge_index, withDataset, withSeed


@withSeed
@withDataset('DIMACS10', 'citationCiteseer')
def test_subgraph(dataset, **kwargs):
    (rowptr, col), num_nodes = dataset, dataset[0].size(0) - 1
    perm = torch.randperm(num_nodes, dtype=rowptr.dtype, device=rowptr.device)
    nodes = perm[:num_nodes // 100]

    t = time.perf_counter()
    for _ in range(10):
        pyg_lib.sampler.subgraph(rowptr, col, nodes)
    print(f'time={time.perf_counter() - t:.6f} seconds')

    edge_index = to_edge_index(rowptr, col)
    from torch_geometric.utils import subgraph

    t = time.perf_counter()
    for _ in range(10):
        subgraph(nodes, edge_index, num_nodes=num_nodes, relabel_nodes=True)
    print(f'time={time.perf_counter() - t:.6f} seconds')


if __name__ == '__main__':
    test_subgraph()
