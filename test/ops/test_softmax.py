import torch

from pyg_lib.ops import softmax
from pyg_lib.testing import onlyCUDA, onlyTriton


@onlyCUDA
@onlyTriton
def test_softmax():
    inputs = torch.randn(8, 5, device='cuda')
    ptr = torch.tensor([0, 3, 8], device='cuda')

    out = softmax(inputs, ptr)
    assert torch.allclose(out[0:3], torch.softmax(inputs[3:8], dim=0))
    assert torch.allclose(out[3:8], torch.softmax(inputs[3:8], dim=0))
