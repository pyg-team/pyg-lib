import argparse

import torch

import pyg_lib

parser = argparse.ArgumentParser()
parser.add_argument('--with_index', action='store_true')
args = parser.parse_args()

a1 = torch.randn(5, 4)
b1 = torch.randn(5, 4)
a_index = torch.tensor([0, 0, 1, 1, 2, 2])
b_index = torch.tensor([0, 0, 1, 1, 2, 2])
grad_out = torch.randn(6 if args.with_index else 5, 4)

a2 = a1.clone()
b2 = b1.clone()

a1.requires_grad_()
b1.requires_grad_()
a2.requires_grad_()
b2.requires_grad_()

if args.with_index:
    out2 = a2[a_index] / b2[b_index]
else:
    out2 = a2 / b2
out2.backward(grad_out)
print(a2.grad)
# print(b2.grad)

if args.with_index:
    out1 = pyg_lib.ops.sampled_div(a1, b1, a_index, b_index)
else:
    out1 = pyg_lib.ops.sampled_div(a1, b1)
out1.backward(grad_out)
print(a1.grad)
# print(b1.grad)
print(torch.allclose(a1.grad, a2.grad))
print(torch.allclose(b1.grad, b2.grad))
