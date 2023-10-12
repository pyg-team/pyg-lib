import torch
import torch.nn.functional as F

import pyg_lib


def softmax_reference_ptr_dim0(src, ptr):
    out = torch.empty_like(src)
    for beg, end in zip(ptr[:-1], ptr[1:]):
        for col in range(src.size(-1)):
            out[beg:end, col] = F.softmax(src[beg:end, col], dim=0)
    return out


def test_softmax_ptr_dim0_autograd():
    src1 = torch.rand((16, 2), requires_grad=True)
    src2 = src1.detach().clone()
    src2.requires_grad = True
    ptr = torch.tensor([0, 7, 15, 16])
    out_grad = torch.randn((16, 2))

    expected_out = softmax_reference_ptr_dim0(src1, ptr)
    out = pyg_lib.ops.softmax(src=src2, ptr=ptr)
    assert torch.allclose(expected_out, out, atol=1e-6)

    expected_out.backward(out_grad)
    out.backward(out_grad)
    assert torch.allclose(src1.grad, src2.grad, atol=1e-6)
