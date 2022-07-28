import torch

import pyg_lib


def test_segment_matmul_autograd():
    inputs = torch.randn((8, 16), requires_grad=True, device='cuda:0')
    ptr = torch.tensor([0, 5, 8]).cuda()
    other = torch.randn((2, 16, 32), requires_grad=True, device='cuda:0')
    out = pyg_lib.ops.segment_matmul(inputs, ptr, other)
    assert out.size() == (8, 32)
    assert out[0:5] == inputs[0:5] @ other[0]
    assert out[5:8] == inputs[5:8] @ other[1]
    out.sum().backward()
    print(out.grad.shape)


def test_grouped_matmul_autograd():
    inputs = [torch.randn(5, 16).cuda(), torch.randn(3, 32).cuda()]
    others = [
        torch.randn((16, 32), requires_grad=True, device='cuda:0'),
        torch.randn((32, 64), requires_grad=True, device='cuda:0')
    ]
    out = pyg_lib.ops.grouped_matmul(inputs, others)
    assert len(outs) == 2
    assert outs[0].size() == (5, 32)
    assert outs[0] == inputs[0] @ others[0]
    assert outs[1].size() == (3, 64)
    assert outs[1] == inputs[1] @ others[1]
    (out[0].sum() + out[1].sum()).backward()
    assert outs[0].grad.size() == (5, 32)
    assert outs[1].grad.size() == (3, 64)


test_segment_matmul_autograd()
test_grouped_matmul_autograd()
