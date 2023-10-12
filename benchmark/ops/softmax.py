import argparse

import torch
import pyg_lib

from time import perf_counter as timestamp
from torch_geometric.utils import scatter, segment


def softmax_reference(src, ptr, dim=0):
    dim = dim + src.dim() if dim < 0 else dim
    size = ([1] * dim) + [-1]
    count = ptr[1:] - ptr[:-1]
    ptr = ptr.view(size)
    src_max = segment(src.detach(), ptr, reduce='max')
    src_max = src_max.repeat_interleave(count, dim=dim)
    out = (src - src_max).exp()
    out_sum = segment(out, ptr, reduce='sum') + 1e-16
    out_sum = out_sum.repeat_interleave(count, dim=dim)

    return out / out_sum


def measure_perf(impl_func, ptr, out_grad, num_warmups, num_steps, backward):
    t_fwd = t_bwd = 0
    for i in range(num_warmups + num_steps):
        src = torch.randn(num_rows, num_heads)
        src.requires_grad = backward

        t_start = timestamp()
        out = impl_func(src=src, ptr=ptr)
        if i >= num_warmups:
            t_fwd += timestamp() - t_start

        if backward:
            t_start = timestamp()
            out.backward(out_grad)
            if i >= num_warmups:
                t_bwd += timestamp() - t_start

    return t_fwd, t_bwd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--num-heads', type=int, default=4)
    args = parser.parse_args()

    num_rows, num_heads = 50000, args.num_heads
    num_warmups, num_steps = 100, 500
    group_size = 100

    ptr = torch.arange(0, num_rows + 1, group_size)
    out_grad = torch.randn(num_rows, num_heads)

    func_args = [ptr, out_grad, num_warmups, num_steps, args.backward]

    t_fwd, t_bwd = measure_perf(softmax_reference, *func_args)
    print(f'Vanilla forward: {t_fwd:.4f}s')
    if args.backward:
        print(f'Vanilla backward: {t_bwd:.4f}s')
    print('=========================')

    t_fwd, t_bwd = measure_perf(pyg_lib.ops.softmax, *func_args)
    print(f'pyg_lib forward:  {t_fwd:.4f}s')
    if args.backward:
        print(f'pyg_lib backward: {t_bwd:.4f}s')

