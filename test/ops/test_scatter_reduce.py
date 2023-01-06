import torch

from pyg_lib.ops import fused_scatter_reduce
from pyg_lib.testing import onlyCUDA, onlyTriton


@onlyCUDA
@onlyTriton
def test_fused_scatter_reduce():
    x = torch.randn(5, 4, device='cuda')
    index = torch.tensor([0, 1, 0, 1, 0], device='cuda')

    out = fused_scatter_reduce(x, index, dim_size=2,
                               reduce_list=['sum', 'mean'])

    assert out.size() == (2, 8)
    assert torch.allclose(out[0, 0:4], x[index == 0].sum(dim=0))
    assert torch.allclose(out[1, 0:4], x[index == 1].sum(dim=0))
    assert torch.allclose(out[0, 4:8], x[index == 0].mean(dim=0))
    assert torch.allclose(out[1, 4:8], x[index == 1].mean(dim=0))


if __name__ == '__main__':  # Benchmarking
    import time

    import torch_scatter

    x = torch.randn(50000, 64, device='cuda')
    index = torch.randint(1000, (x.size(0), ), device='cuda')

    num_warmups = 1000
    num_steps = 10000

    for i in range(num_warmups + num_steps):
        if i == num_warmups:
            torch.cuda.synchronize()
            t = time.perf_counter()
        out_fused = fused_scatter_reduce(x, index, dim_size=1000,
                                         reduce_list=['sum', 'mean'])
    torch.cuda.synchronize()
    t = time.perf_counter() - t
    print(f'  Fused implementation: {t:.4f} seconds')

    for i in range(num_warmups + num_steps):
        if i == num_warmups:
            torch.cuda.synchronize()
            t = time.perf_counter()
        out1 = torch_scatter.scatter_add(x, index, dim_size=1000, dim=0)
        out2 = torch_scatter.scatter_mean(x, index, dim_size=1000, dim=0)
        out_vanilla = torch.cat([out1, out2], dim=-1)
    torch.cuda.synchronize()
    t = time.perf_counter() - t
    print(f'Vanilla implementation: {t:.4f} seconds')

    assert torch.allclose(out_fused, out_vanilla, atol=1e-5)
