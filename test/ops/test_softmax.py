import pytest
import torch

import pyg_lib


def ptr2index(ptr):
    group_sizes = ptr[1:] - ptr[:-1]
    return torch.repeat_interleave(
        torch.arange(0, group_sizes.numel(), dtype=group_sizes.dtype),
        group_sizes)


def broadcast(src, ref, dim):
    size = ((1, ) * dim) + (-1, ) + ((1, ) * (ref.dim() - dim - 1))
    return src.view(size).expand_as(ref)


def softmax_reference(src, ptr, dim):
    index = ptr2index(ptr)
    N = int(index.max()) + 1 if index.numel() > 0 else 0
    size = src.size()[:dim] + (N, ) + src.size()[dim + 1:]
    src_max = src.detach().new_zeros(size).scatter_reduce_(
        dim, broadcast(index, src, dim), src, reduce='amax',
        include_self=False)
    out = src - src_max.index_select(dim, index)
    out = out.exp()
    out_sum = out.new_zeros(size).scatter_add_(dim, broadcast(index, out, dim),
                                               out)
    out_sum = out_sum.index_select(dim, index)

    return out / out_sum


@pytest.mark.skipif(
    torch.__version__.startswith('2.4.0'),
    reason="https://github.com/pytorch/pytorch/issues/130619",
)
@pytest.mark.parametrize('dim', [0, 1, 2])
def test_softmax_csr_autograd(dim):
    sizes = (16, 32, 64)
    src1 = torch.rand(sizes, requires_grad=True)
    src2 = src1.detach().clone()
    src2.requires_grad = True
    dim_size = sizes[dim]
    ptr = torch.tensor([0, 1, 4, 5, dim_size - 1, dim_size])
    out_grad = torch.randn(sizes)

    expected_out = softmax_reference(src1, ptr, dim)
    out = pyg_lib.ops.softmax_csr(src2, ptr, dim)
    assert torch.allclose(expected_out, out, atol=1e-6)

    expected_out.backward(out_grad)
    out.backward(out_grad)
    assert torch.allclose(src1.grad, src2.grad, atol=1e-6)
