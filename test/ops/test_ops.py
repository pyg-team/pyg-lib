import pytest
import torch

import pyg_lib

#DEVICE_STRS = ['cpu'] # cpu has enormous errors or segfaults, see Issue 119
DEVICE_STRS = []
if torch.cuda.is_available():
    DEVICE_STRS.append('cuda:0')


def assert_close_enough(x, y, tol=6e-3):
    # TODO Rishi: Work w/ Cutlass to lower the high error (as large as ~5e-3)
    assert ((x - y).abs().max() <= tol), 'Max Abs Err: ' + str(
        (x - y).abs().max()) + ', Tolerace: ' + str(tol)


@pytest.mark.parametrize('device_str', DEVICE_STRS)
def test_segment_matmul_autograd(device_str):
    inputs = torch.randn((8, 16), requires_grad=True, device=device_str)
    ptr = torch.tensor([0, 5, 8]).to(torch.device(device_str))
    other = torch.randn((2, 16, 32), requires_grad=True, device=device_str)
    out = pyg_lib.ops.segment_matmul(inputs, ptr, other)
    assert out.size() == (8, 32)
    tol = 1e-7 if device_str == 'cpu' else 6e-3
    assert_close_enough(out[0:5], inputs[0:5] @ other[0], tol)
    assert_close_enough(out[5:8], inputs[5:8] @ other[1], tol)
    out.sum().backward()
    assert other.grad.shape == other.shape
    assert inputs.grad.shape == inputs.shape

    print('test_segment_matmul_autograd passed!')


@pytest.mark.parametrize('device_str', DEVICE_STRS)
def test_grouped_matmul_autograd(device_str):
    device = torch.device(device_str)
    inputs = [torch.randn(5, 16).to(device), torch.randn(3, 32).to(device)]
    others = [
        torch.randn((16, 32), requires_grad=True, device=device_str),
        torch.randn((32, 64), requires_grad=True, device=device_str)
    ]
    outs = pyg_lib.ops.grouped_matmul(inputs, others)
    assert len(outs) == 2
    assert outs[0].size() == (5, 32)
    assert outs[1].size() == (3, 64)
    tol = 1e-7 if device_str == 'cpu' else 6e-3
    assert_close_enough(outs[0], inputs[0] @ others[0], tol)
    assert_close_enough(outs[1], inputs[1] @ others[1], tol)
    (outs[0].sum() + outs[1].sum()).backward()
    assert outs[0].grad.size() == (5, 32)
    assert outs[1].grad.size() == (3, 64)
    print('test_grouped_matmul_autograd passed!')


if __name__ == '__main__':
    for device_str in DEVICE_STRS:
        # test_segment_matmul_autograd(device_str)
        test_grouped_matmul_autograd(device_str)
