import argparse
import time

import torch

import pyg_lib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    num_nodes, num_edges, num_feats = 10000, 50000, 64

    num_warmups, num_steps = 500, 1000
    if args.device == 'cpu':
        num_warmups, num_steps = num_warmups // 10, num_steps // 10

    a_index = torch.randint(0, num_nodes, (num_edges, ), device=args.device)
    b_index = torch.randint(0, num_nodes, (num_edges, ), device=args.device)
    out_grad = torch.randn(num_edges, num_feats, device=args.device)

    for fn in ['add', 'sub', 'mul', 'div']:
        print(f'Function: {fn}')
        print('=========================')

        op = getattr(torch, fn)
        t_forward = t_backward = 0
        for i in range(num_warmups + num_steps):
            a = torch.randn(num_nodes, num_feats, device=args.device)
            b = torch.randn(num_nodes, num_feats, device=args.device)
            if args.backward:
                a.requires_grad_(True)
                b.requires_grad_(True)

            torch.cuda.synchronize()
            t_start = time.perf_counter()

            out = op(a[a_index], b[b_index])

            torch.cuda.synchronize()
            if i >= num_warmups:
                t_forward += time.perf_counter() - t_start

            if args.backward:
                t_start = time.perf_counter()
                out.backward(out_grad)

                torch.cuda.synchronize()
                if i >= num_warmups:
                    t_backward += time.perf_counter() - t_start

        print(f'Vanilla forward:  {t_forward:.4f}s')
        if args.backward:
            print(f'Vanilla backward: {t_backward:.4f}s')
        print('=========================')

        op = getattr(pyg_lib.ops, f'sampled_{fn}')
        t_forward = t_backward = 0
        for i in range(num_warmups + num_steps):
            a = torch.randn(num_nodes, num_feats, device=args.device)
            b = torch.randn(num_nodes, num_feats, device=args.device)
            if args.backward:
                a.requires_grad_(True)
                b.requires_grad_(True)

            torch.cuda.synchronize()
            t_start = time.perf_counter()

            out = op(a, b, a_index, b_index)

            torch.cuda.synchronize()
            if i >= num_warmups:
                t_forward += time.perf_counter() - t_start

            if args.backward:
                t_start = time.perf_counter()
                out.backward(out_grad)

                torch.cuda.synchronize()
                if i >= num_warmups:
                    t_backward += time.perf_counter() - t_start

        print(f'pyg_lib forward:  {t_forward:.4f}s')
        if args.backward:
            print(f'pyg_lib backward: {t_backward:.4f}s')

        print()
