import torch

from pyg_lib.ops import broadcast_sub
from pyg_lib.testing import onlyCUDA, onlyTriton


@onlyCUDA
@onlyTriton
def test_broadcast_sub():
    inputs = torch.randn(100, 16, device='cuda')
    other = torch.randn(10, 16, device='cuda')
    index = torch.randint(0, 10, (100, ), device='cuda')

    out = broadcast_sub(inputs, other, index)
    assert out.size() == inputs.size()
    assert torch.allclose(out, inputs - other[index])
