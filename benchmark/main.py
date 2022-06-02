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
    print(time.perf_counter() - t)

    edge_index = to_edge_index(rowptr, col)
    from torch_geometric.utils import subgraph

    t = time.perf_counter()
    for _ in range(10):
        subgraph(nodes, edge_index, num_nodes=num_nodes, relabel_nodes=True)
    print(time.perf_counter() - t)


def test_segment_matmul():
    num_types = 100
    num_nodes = 10000
    feat = 128

    inputs = torch.randn(num_types, num_nodes, feat, device='cuda')
    weight = torch.randn(num_types, feat, feat, device='cuda')
    out = torch.empty(num_types, num_nodes, feat, device='cuda')
    ptr = torch.arange(num_types + 1)

    for i in range(1, 1001):
        if i == 100:
            t = time.perf_counter()
        torch.cuda.synchronize()
        torch.ops.pyg.segment_matmul(inputs, ptr, weight, out)
        torch.cuda.synchronize()
    print(time.perf_counter() - t)

    seglen = torch.zeros(inputs.size(0), dtype=torch.long,
                         device='cpu') + inputs.size(1)
    import dgl
    for i in range(1, 1001):
        if i == 100:
            t = time.perf_counter()
        torch.cuda.synchronize()
        dgl.ops.segment_mm(inputs.view(-1, feat), weight, seglen)
        torch.cuda.synchronize()
    print(time.perf_counter() - t)

    for i in range(1, 1001):
        if i == 100:
            t = time.perf_counter()
        torch.cuda.synchronize()
        out = torch.empty_like(inputs)
        for j in range(inputs.size(0)):
            out[j] = inputs[j] @ weight[j]
        torch.cuda.synchronize()
    print(time.perf_counter() - t)

    pass


if __name__ == '__main__':
    # test_subgraph()
    test_segment_matmul()
