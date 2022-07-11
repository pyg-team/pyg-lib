import torch

def test_grouped_matmul():
    from pyg_lib.ops import grouped_matmul

    # compute forward
    inputs = [torch.randn(5, 16), torch.randn(3, 32)]
    others = [torch.randn((16, 32), requires_grad=True), torch.randn((32, 64), requires_grad=True)]
    outs = grouped_matmul(inputs, others)
    # check forward correctness
    assert len(outs) == 2
    assert outs[0].size() == (5, 32)
    assert outs[0] == inputs[0] @ others[0]
    assert outs[1].size() == (3, 64)
    assert outs[1] == inputs[1] @ others[1]
    # compute backward
    (outs[0] + outs[1]).sum().backward()

    # check correctness of backward
    assert others[0].grad == outs[0].grad @ others[0].T
    assert others[1].grad == outs[1].grad @ others[1].T


def test_segment_matmul():
    from pyg_lib.ops import segment_matmul
    # compute forward
    inputs = torch.randn(8, 16)
    ptr = torch.tensor([0, 5, 8])
    other = torch.randn((2, 16, 32), requires_grad=True)
    out = segment_matmul(inputs, ptr, other)
    # check forward correctness
    assert out.size() == (8, 32)
    assert out[0:5] == inputs[0:5] @ other[0]
    assert out[5:8] == inputs[5:8] @ other[1]

    # compute backward
    out.backward(torch.ones_like(out))

    # check correctness of backward
    assert other.grad[0:5] == out[0:5].grad @ other[0:5].T
    assert other.grad[5:8] == out[5:8].grad @ other[5:8].T





if __name__ == 'main':
    test_grouped_matmul()
    test_segment_matmul()