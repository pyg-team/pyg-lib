import os

import pytest
import torch

import pyg_lib
from pyg_lib._compile import _WITH_PT24
from pyg_lib.testing import withCUDA

os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision('highest')  # Enforce FP32


@withCUDA
@pytest.mark.parametrize('dtype', [torch.float, torch.bfloat16])
def test_segment_matmul_autograd(dtype, device):
    if device.type == 'cuda' and dtype == torch.bfloat16:
        pytest.skip('CUDA does not support bfloat16')

    inputs = torch.randn((8, 16), requires_grad=True, device=device,
                         dtype=dtype)
    ptr = torch.tensor([0, 5, 8]).to(torch.device(device))
    other = torch.randn((2, 16, 32), requires_grad=True, device=device,
                        dtype=dtype)
    bias = torch.randn((2, 32), requires_grad=True, device=device, dtype=dtype)
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
@pytest.mark.parametrize('requires_grad', [False, True])
@pytest.mark.skipif(not _WITH_PT24, reason='PyTorch 2.4.0 is required')
def test_segment_matmul_opcheck(device, requires_grad):
    if requires_grad:
        pytest.skip('TODO: Support requires_grad=True')

    from torch.library import opcheck

    dtype = torch.float32
    inputs = torch.randn((8, 16), requires_grad=requires_grad, device=device,
                         dtype=dtype)
    ptr = torch.tensor([0, 5, 8], device=device)
    other = torch.randn((2, 16, 32), requires_grad=requires_grad,
                        device=device, dtype=dtype)
    opcheck(torch.ops.pyg.segment_matmul, (inputs, ptr, other),
            test_utils="test_schema")
    opcheck(torch.ops.pyg.segment_matmul, (inputs, ptr, other),
            test_utils="test_autograd_registration")
    opcheck(torch.ops.pyg.segment_matmul, (inputs, ptr, other),
            test_utils="test_faketensor")
    opcheck(torch.ops.pyg.segment_matmul, (inputs, ptr, other),
            test_utils="test_aot_dispatch_static")
    # TODO(akihironitta): Support dynamic shapes
    # opcheck(torch.ops.pyg.segment_matmul, (inputs, ptr, other),
    #         test_utils="test_aot_dispatch_dynamic")


@withCUDA
@pytest.mark.parametrize('dtype', [torch.float, torch.bfloat16])
@pytest.mark.parametrize('transposed', [True, False])
def test_grouped_matmul_autograd(dtype, transposed, device):
    if device.type == 'cuda' and dtype == torch.bfloat16:
        pytest.skip('CUDA does not support bfloat16')

    inputs = [
        torch.randn(5, 16, device=device, requires_grad=True),
        torch.randn(6, 9, device=device, requires_grad=True),
        torch.randn(3, 32, device=device, requires_grad=True),
    ]
    if transposed:
        others_origin = [
            torch.randn(48, 16, device=device, requires_grad=True),
            torch.randn(42, 9, device=device, requires_grad=True),
            torch.randn(64, 32, device=device, requires_grad=True),
        ]
        others = [other.t() for other in others_origin]
    else:
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
        assert torch.allclose(outs[i], expected, atol=1e-4)

    sum([out.sum() for out in outs]).backward()
    for i in range(len(outs)):
        if transposed:
            assert others_origin[i].grad.size() == others_origin[i].size()
        else:
            assert others[i].grad.size() == others[i].size()
