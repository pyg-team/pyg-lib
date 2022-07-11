import pytest
import torch


def test_grouped_matmul():
    from pyg_lib.ops import grouped_matmul
    # compute
    inputs = [torch.randn(5, 16), torch.randn(3, 32), torch.randn(4, 10)]
    others = [torch.randn(16, 32), torch.randn(32, 64), torch.randn(10, 20)]
    outs = pyg_lib.segment.grouped_matmul(inputs, others)
    # check correctness
    assert len(outs) == 2
    assert outs[0].size() == (5, 32)
    assert outs[0] == inputs[0] @ others[0]
    assert outs[1].size() == (3, 64)
    assert outs[1] == inputs[1] @ others[1]
    assert outs[2].size() == (4, 20)
    assert outs[2] == inputs[2] @ others[2]

def test_segment_matmul():
    from pyg_lib.ops import segment_matmul
    # compute
    inputs = torch.randn(8, 16)
    ptr = torch.tensor([0, 5, 8])
    other = torch.randn(2, 16, 32)
    out = pyg_lib.segment.segment_matmul(inputs, ptr, other)
    # check correctness
    assert out.size() == (8, 32)
    assert out[0:5] == inputs[0:5] @ other[0]
    assert out[5:8] == inputs[5:8] @ other[1]


