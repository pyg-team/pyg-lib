import torch

from pyg_lib.ops import softmax
from pyg_lib.testing import onlyCUDA, onlyTriton


@onlyCUDA
@onlyTriton
def test_softmax():
    inputs = torch.randn(8, 5, device='cuda')
    ptr = torch.tensor([0, 3, 8], device='cuda')

    print()
    a = inputs[0:3]
    a = (a - a.max(dim=0, keepdim=True)[0]).exp()
    a = a / a.sum(dim=0, keepdim=True)
    print(a)
    b = inputs[3:8]
    b = (b - b.max(dim=0, keepdim=True)[0]).exp()
    b = b / b.sum(dim=0, keepdim=True)
    print(b)

    out = softmax(inputs, ptr)
    print()
    # print((inputs[0:3] - inputs[0:3].max(dim=0, keepdim=True)[0]).exp())
    # print((inputs[3:8] - inputs[3:8].max(dim=0, keepdim=True)[0]).exp())
    print()
    print(out)
    print()
    # print(out[0:3])
    print(torch.softmax(inputs[0:3], dim=0))
    # assert torch.allclose(out[0:3], torch.softmax(inputs[0:3], dim=0))
    # print(out[3:8])
    print(torch.softmax(inputs[3:8], dim=0))
    # assert torch.allclose(out[3:8], torch.softmax(inputs[3:8], dim=0))
