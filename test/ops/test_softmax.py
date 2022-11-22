import torch

from pyg_lib.ops import softmax
from pyg_lib.testing import onlyCUDA, onlyTriton


@onlyCUDA
@onlyTriton
def test_softmax():
    torch.manual_seed(12345)
    inputs = torch.tensor([
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5],
    ], device='cuda', dtype=torch.float)
    # inputs = torch.randn(8, 5, device='cuda')
    ptr = torch.tensor([0, 2, 8], device='cuda')

    print()
    # print(inputs)

    out = softmax(inputs, ptr)
    print(out)
    # print(torch.allclose(out, inputs))

    a = inputs[0:2]
    # a = a - a.max(dim=0, keepdim=True)[0]
    a = a.exp()
    a = a / a.sum(dim=0, keepdim=True)
    print(a)
    # print(inputs[0:3].softmax(dim=0))
    # print(inputs[3:8].softmax(dim=0))

    # assert torch.allclose(out[0:3], torch.softmax(inputs[3:8], dim=0))
    # assert torch.allclose(out[3:8], torch.softmax(inputs[3:8], dim=0))
