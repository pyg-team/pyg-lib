import pytest
import torch
import os
import pyg_lib

DEVICE_STRS = ['cpu', 'cuda:0']
major_vers, minor_vers = str(torch.__version__).split('.')[:2]
print(major_vers, minor_vers)
test_group_matmul = int(major_vers) >= 2 or int(minor_vers) >= 14
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

def assert_close_enough(x, y, tol=1e-5):
    assert ((x - y).abs().max() <= tol), 'Max Abs Err: ' + str(float(
        (x - y).abs().max())) + ', Tolerance: ' + str(tol)


@pytest.mark.parametrize('device_str', DEVICE_STRS)
def test_segment_matmul_autograd(device_str):
    inputs = torch.randn((8, 16), requires_grad=True, device=device_str)
    ptr = torch.tensor([0, 5, 8]).to(torch.device(device_str))
    other = torch.randn((2, 16, 32), requires_grad=True, device=device_str)
    out = pyg_lib.ops.segment_matmul(inputs, ptr, other)
    assert out.shape == (inputs.shape[0], other.shape[-1])
    assert_close_enough(out[0:ptr[1]], inputs[0:ptr[1]] @ other[0])
    assert_close_enough(out[ptr[1]:ptr[2]], inputs[ptr[1]:ptr[2]] @ other[1])
    out.sum().backward()
    assert other.grad.shape == other.shape
    assert inputs.grad.shape == inputs.shape

@pytest.mark.skipif(not test_group_matmul, reason="grouped_matmul requires torch >= 1.14")
@pytest.mark.parametrize('device_str', DEVICE_STRS)
def test_grouped_matmul_autograd(device_str):
    device = torch.device(device_str)
    inputs = [torch.randn(5, 16).to(device), torch.randn(6, 9).to(device), torch.randn(3, 32).to(device)]
    others = [
        torch.randn((16, 48), requires_grad=True, device=device_str),
        torch.randn((9, 42), requires_grad=True, device=device_str),
        torch.randn((32, 64), requires_grad=True, device=device_str)
    ]
    outs = pyg_lib.ops.grouped_matmul(inputs, others)
    assert len(outs) == len(inputs)
    for i in range(len(outs)):
        assert outs[i].size() == (inputs[i].shape[0], others[i].shape[-1])
        assert_close_enough(outs[i], inputs[i] @ others[i])
    
    sum([out.sum() for out in outs]).backward()
    for i in range(len(outs)):
        assert others[i].grad.shape == others[i].shape