import argparse

import torch
from torch.utils.benchmark import Compare, Timer

import pyg_lib

try:
    import torch_spline_conv
    HAS_TORCH_SPLINE_CONV = True
except ImportError:
    HAS_TORCH_SPLINE_CONV = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    bench_original = False
    if HAS_TORCH_SPLINE_CONV:
        try:
            p = torch.rand(1, 1, device=args.device)
            ks = torch.tensor([3], dtype=torch.long, device=args.device)
            ios = torch.tensor([1], dtype=torch.uint8, device=args.device)
            torch_spline_conv.spline_basis(p, ks, ios, 1)
            bench_original = True
        except RuntimeError:
            pass

    results = []

    # --- spline_basis ---
    for degree in [1, 2, 3]:
        E, D = 10000, 3
        pseudo = torch.rand(E, D, dtype=torch.float, device=args.device)
        kernel_size = torch.tensor([5] * D, dtype=torch.long,
                                   device=args.device)
        is_open_spline = torch.tensor([1] * D, dtype=torch.uint8,
                                      device=args.device)
        label = f'spline_basis (degree={degree}, E={E}, D={D})'

        if bench_original:
            t = Timer(
                stmt='torch_spline_conv.spline_basis(pseudo, kernel_size, '
                'is_open_spline, degree)',
                globals={
                    'torch_spline_conv': torch_spline_conv,
                    'pseudo': pseudo,
                    'kernel_size': kernel_size,
                    'is_open_spline': is_open_spline,
                    'degree': degree,
                },
                label=label,
                sub_label='fwd',
                description='torch_spline_conv',
            )
            results.append(t.blocked_autorange())

        t = Timer(
            stmt='pyg_lib.ops.spline_basis(pseudo, kernel_size, '
            'is_open_spline, degree)',
            globals={
                'pyg_lib': pyg_lib,
                'pseudo': pseudo,
                'kernel_size': kernel_size,
                'is_open_spline': is_open_spline,
                'degree': degree,
            },
            label=label,
            sub_label='fwd',
            description='pyg_lib',
        )
        results.append(t.blocked_autorange())

        if bench_original:
            pseudo_orig = pseudo.clone().requires_grad_(True)
            t = Timer(
                stmt='torch_spline_conv.spline_basis(pseudo, kernel_size, '
                'is_open_spline, degree)[0].sum().backward()',
                globals={
                    'torch_spline_conv': torch_spline_conv,
                    'pseudo': pseudo_orig,
                    'kernel_size': kernel_size,
                    'is_open_spline': is_open_spline,
                    'degree': degree,
                },
                label=label,
                sub_label='fwd+bwd',
                description='torch_spline_conv',
            )
            results.append(t.blocked_autorange())

        pseudo_new = pseudo.clone().requires_grad_(True)
        t = Timer(
            stmt='pyg_lib.ops.spline_basis(pseudo, kernel_size, '
            'is_open_spline, degree)[0].sum().backward()',
            globals={
                'pyg_lib': pyg_lib,
                'pseudo': pseudo_new,
                'kernel_size': kernel_size,
                'is_open_spline': is_open_spline,
                'degree': degree,
            },
            label=label,
            sub_label='fwd+bwd',
            description='pyg_lib',
        )
        results.append(t.blocked_autorange())

    # --- spline_weighting ---
    E, M_in, M_out = 10000, 8, 16
    K, S = 125, 8

    x = torch.randn(E, M_in, dtype=torch.float, device=args.device)
    weight = torch.randn(K, M_in, M_out, dtype=torch.float, device=args.device)
    basis = torch.rand(E, S, dtype=torch.float, device=args.device)
    weight_index = torch.randint(0, K, (E, S), dtype=torch.long,
                                 device=args.device)
    label = f'spline_weighting (E={E}, M_in={M_in}, M_out={M_out}, K={K})'

    if bench_original:
        t = Timer(
            stmt='torch_spline_conv.spline_weighting(x, weight, basis, '
            'weight_index)',
            globals={
                'torch_spline_conv': torch_spline_conv,
                'x': x,
                'weight': weight,
                'basis': basis,
                'weight_index': weight_index,
            },
            label=label,
            sub_label='fwd',
            description='torch_spline_conv',
        )
        results.append(t.blocked_autorange())

    t = Timer(
        stmt='pyg_lib.ops.spline_weighting(x, weight, basis, weight_index)',
        globals={
            'pyg_lib': pyg_lib,
            'x': x,
            'weight': weight,
            'basis': basis,
            'weight_index': weight_index,
        },
        label=label,
        sub_label='fwd',
        description='pyg_lib',
    )
    results.append(t.blocked_autorange())

    if bench_original:
        x_orig = x.clone().requires_grad_(True)
        w_orig = weight.clone().requires_grad_(True)
        b_orig = basis.clone().requires_grad_(True)
        t = Timer(
            stmt='torch_spline_conv.spline_weighting(x, weight, basis, '
            'weight_index).sum().backward()',
            globals={
                'torch_spline_conv': torch_spline_conv,
                'x': x_orig,
                'weight': w_orig,
                'basis': b_orig,
                'weight_index': weight_index,
            },
            label=label,
            sub_label='fwd+bwd',
            description='torch_spline_conv',
        )
        results.append(t.blocked_autorange())

    x_new = x.clone().requires_grad_(True)
    w_new = weight.clone().requires_grad_(True)
    b_new = basis.clone().requires_grad_(True)
    t = Timer(
        stmt='pyg_lib.ops.spline_weighting(x, weight, basis, '
        'weight_index).sum().backward()',
        globals={
            'pyg_lib': pyg_lib,
            'x': x_new,
            'weight': w_new,
            'basis': b_new,
            'weight_index': weight_index,
        },
        label=label,
        sub_label='fwd+bwd',
        description='pyg_lib',
    )
    results.append(t.blocked_autorange())

    compare = Compare(results)
    compare.trim_significant_figures()
    compare.colorize(rowwise=True)
    compare.print()
