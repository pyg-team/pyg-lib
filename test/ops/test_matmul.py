import os

import torch

import pyg_lib
from pyg_lib.testing import withCUDA

os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision('highest')  # Enforce FP32


@withCUDA
def test_segment_matmul_autograd(device):
    inputs = torch.randn((8, 16), requires_grad=True, device=device)
    ptr = torch.tensor([0, 5, 8]).to(torch.device(device))
    other = torch.randn((2, 16, 32), requires_grad=True, device=device)
    bias = torch.randn((2, 32), requires_grad=True, device=device)
    out = pyg_lib.ops.segment_matmul(inputs, ptr, other, bias)
    assert out.size() == (8, 32)

    out1 = inputs[ptr[0]:ptr[1]] @ other[0] + bias[0]
    assert torch.allclose(out[ptr[0]:ptr[1]], out1, atol=1e-6)

    out2 = inputs[ptr[1]:ptr[2]] @ other[1] + bias[1]
    assert torch.allclose(out[ptr[1]:ptr[2]], out2, atol=1e-6)

    out.mean().backward()
    assert other.grad.size() == other.size()
    assert inputs.grad.size() == inputs.size()


@withCUDA
def test_grouped_matmul_autograd(device):
    inputs = [
        torch.randn(5, 16, device=device),
        torch.randn(6, 9, device=device),
        torch.randn(3, 32, device=device),
    ]
    others = [
        torch.randn(16, 48, device=device, requires_grad=True),
        torch.randn(9, 42, device=device, requires_grad=True),
        torch.randn(32, 64, device=device, requires_grad=True),
    ]

    biases = [
        torch.randn(48, device=device, requires_grad=True),
        torch.randn(42, device=device, requires_grad=True),
        torch.randn(64, device=device, requires_grad=True),
    ]

    outs = pyg_lib.ops.grouped_matmul(inputs, others, biases)
    assert len(outs) == len(inputs)

    for i in range(len(outs)):
        assert outs[i].size() == (inputs[i].size(0), others[i].size(-1))
        expected = inputs[i] @ others[i] + biases[i]
        assert torch.allclose(outs[i], expected, atol=1e-6)

    sum([out.sum() for out in outs]).backward()
    for i in range(len(outs)):
        assert others[i].grad.size() == others[i].size()
