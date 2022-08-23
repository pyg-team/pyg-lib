import torch
from torch_sparse import SparseTensor
import pyg_lib

neighbor_sample = pyg_lib.sampler.neighbor_sample

def test_neighbor_sample():                         # row, col
    adj = SparseTensor.from_edge_index(torch.tensor([[0], [1]]))
    rowptr, col, _ = adj.csr()

    # Sampling in a non-directed way should not sample in wrong direction:
    out = neighbor_sample(rowptr, col, torch.tensor([1]), [1], False, False, False, True)
    assert out[0].tolist() == []
    assert out[1].tolist() == []
    assert out[2].tolist() == [1]

    # Sampling should work:
    out = neighbor_sample(rowptr, col, torch.tensor([0]), [1], False, False, False, True)
    assert out[0].tolist() == [0]
    assert out[1].tolist() == [1]
    assert out[2].tolist() == [0, 1]

    # Sampling with more hops:
    out = neighbor_sample(rowptr, col, torch.tensor([0]), [1, 1], False, False, False, True)
    assert out[0].tolist() == [0]
    assert out[1].tolist() == [1]
    assert out[2].tolist() == [0, 1]


def test_neighbor_sample_seed():
    rowptr = torch.tensor([0, 3, 5])
    col = torch.tensor([0, 1, 2, 0, 1, 0, 2])
    input_nodes = torch.tensor([0, 1])

    torch.manual_seed(42)
    out1 = neighbor_sample(rowptr, col, input_nodes, [1], True, False, False, True)

    torch.manual_seed(42)
    out2 = neighbor_sample(rowptr, col, input_nodes, [1], True, False, False, True)

    for data1, data2 in zip(out1, out2):
        assert data1.tolist() == data2.tolist()
