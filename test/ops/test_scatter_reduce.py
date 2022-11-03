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
