from typing import List

import torch
from torch import Tensor


class SegmentMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, ptr, other):
        assert 'cuda' in input_tensor.device and 'cuda' in ptr.device and 'cuda' in other.device, 'Only CUDA Tensors supported'
        ctx.save_for_backward(input_tensor, ptr, other)
        return torch.ops.pyg.segment_matmul_kern(input_tensor, ptr, other)

    @staticmethod
    def backward(ctx, gradout):
        input_tensor, ptr, other = ctx.saved_tensors
        input_grad, other_grad = None, None
        if input_tensor.requires_grad:
            input_grad = torch.ops.pyg.segment_matmul_kern(
                gradout, ptr, other.T)
        if other.requires_grad:
            sizes = (ptr[1:] - ptr[:-1]).tolist()
            split_input_T = torch.split(input_tensor.T, sizes, dim=1)
            grad_out_split = torch.split(gradout, sizes, dim=0)
            other_grad = torch.stack(
                torch.ops.pyg.grouped_matmul_kern(split_input_T,
                                                  grad_out_split))

        return input_grad, None, other_grad


class GroupedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, others):
        assert all([
            'cuda' in inputs[i].device and 'cuda' in others[i].device
            for i in range(len(inputs))
        ])
        ctx.save_for_backward(inputs, others)
        return torch.ops.pyg.grouped_matmul_kern(inputs, others)

    @staticmethod
    def backward(ctx, gradouts):
        inputs, others = ctx.saved_tensors
        inputs_grads, others_grads = None, None
        if any([i.requires_grad for i in inputs]):
            pass
        if any([i.requires_grad for i in others]):
            pass
        return inputs_grads, others_grads


def grouped_matmul(inputs: List[Tensor], others: List[Tensor]) -> List[Tensor]:
    r"""Performs dense-dense matrix multiplication according to groups,
    utilizing dedicated kernels that effectively parallelize over groups.

    .. code-block:: python
        inputs = [torch.randn(5, 16), torch.randn(3, 32)]
        others = [torch.randn(16, 32), torch.randn(32, 64)]

        outs = pyg_lib.ops.grouped_matmul(inputs, others)
        assert len(outs) == 2
        assert outs[0].size() == (5, 32)
        assert outs[0] == inputs[0] @ others[0]
        assert outs[1].size() == (3, 64)
        assert outs[1] == inputs[1] @ others[1]

    Args:
        inputs (List[torch.Tensor]): List of left operand 2D matrices of shapes
            :obj:`[N_i, K_i]`.
        others (List[torch.Tensor]): List of right operand 2D matrices of
            shapes :obj:`[K_i, M_i]`.

    Returns:
        List[torch.Tensor]: List of 2D output matrices of shapes
            :obj:`[N_i, M_i]`.
    """
    return GroupedMatmul.apply(inputs, others)


def segment_matmul(inputs: Tensor, ptr: Tensor, other: Tensor) -> Tensor:
    r"""Performs dense-dense matrix multiplication according to segments along
    the first dimension of :obj:`inputs` as given by :obj:`ptr`, utilizing
    dedicated kernels that effectively parallelize over groups.

    .. code-block:: python
        inputs = torch.randn(8, 16)
        ptr = torch.tensor([0, 5, 8])
        other = torch.randn(2, 16, 32)

        out = pyg_lib.ops.segment_matmul(inputs, ptr, other)
        assert out.size() == (8, 32)
        assert out[0:5] == inputs[0:5] @ other[0]
        assert out[5:8] == inputs[5:8] @ other[1]

    Args:
        input (torch.Tensor): The left operand 2D matrix of shape
            :obj:`[N, K]`.
        ptr (torch.Tensor): Compressed vector of shape :obj:`[B + 1]`, holding
            the boundaries of segments.
            For best performance, given as a CPU tensor.
        other (torch.Tensor): The right operand 3D tensor of shape
            :obj:`[B, K, M]`.

    Returns:
        torch.Tensor: The 2D output matrix of shape :obj:`[N, M]`.
    """
    return SegmentMatmul.apply(inputs, ptr, other)


__all__ = [
    'grouped_matmul',
    'segment_matmul',
]
