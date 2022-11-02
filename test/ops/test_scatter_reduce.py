import torch

from pyg_lib.ops import fused_scatter_reduce
from pyg_lib.testing import onlyCUDA, onlyTriton


@onlyCUDA
@onlyTriton
def test_fused_scatter_reduce():
    x = torch.randn(5, 4, device='cuda')
    index = torch.tensor([0, 1, 0, 1, 0], device='cuda')

    out = fused_scatter_reduce(x, index, dim_size=2, reduce=['sum', 'max'])
    assert out.size() == (2, 8)
