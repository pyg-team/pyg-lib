from typing import List

import torch
from torch import Tensor


class SegmentMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, ptr, other):
        assert inputs.is_cuda
        assert ptr.is_cuda
        assert other.is_cuda
        ctx.save_for_backward(inputs, ptr, other)
        return torch.ops.pyg.cuda_segment_matmul(inputs, ptr, other)

    @staticmethod
    def backward(ctx, out_grad):
        inputs, ptr, other = ctx.saved_tensors

        input_grad = None
        if inputs.requires_grad:
            input_grad = torch.ops.pyg.cuda_segment_matmul(
                out_grad, ptr, torch.transpose(other, -2, -1))

        other_grad = None, None
        if other.requires_grad:
            sizes = (ptr[1:] - ptr[:-1]).tolist()
            inputs_t = inputs.transpose(-2, -1).split(sizes, dim=1)
            outs_grad = out_grad.split(sizes, dim=0)
            others_grad = []
            # Considering GPU utilization, this is actually preferred over grouped matmul
            for i in range(len(inputs_t)):
                others_grad.append(inputs_t @ outs_grad)
            other_grad = torch.stack(others_grad, dim=0)

        return input_grad, None, other_grad


class GroupedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: List[Tensor], others: List[Tensor]):
        for x, other in zip(inputs, others):
            assert x.is_cuda
            assert other.is_cuda
        ctx.save_for_backward(inputs, others)
        outs = torch.ops.pyg.cuda_grouped_matmul(inputs, others)

        # NOTE Autograd doesnt set out[i].requires_grad = True automatically
        for i in range(len(outs)):
            outs[i].requires_grad = True

        return outs

    @staticmethod
    def backward(ctx, outs_grad: List[Tensor]):
        inputs, others = ctx.saved_tensors

        inputs_grad = None
        if all([x.requires_grad for x in inputs]):
            for i in range(len(others)):
                others[i] = others[i].t()
            inputs_grad = torch.ops.pyg.cuda_grouped_matmul(outs_grad, others)

        others_grad = None
        if all([other.requires_grad for other in others]):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].t()
            others_grad = torch.ops.pyg.cuda_grouped_matmul(inputs, outs_grad)

        return inputs_grad, others_grad


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
