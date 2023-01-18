import torch

import pyg_lib

torch.manual_seed(1234)


def test_index_sort():
    input = torch.randint(low=0, high=1024, size=(1000000, ))
    ref_sorted_input, ref_indices = torch.sort(input, stable=True)
    sorted_input, indices = pyg_lib.ops.index_sort(input)
    assert torch.all(ref_sorted_input == sorted_input)
    assert torch.all(ref_indices == indices)


def test_index_sort_negative():
    input = torch.randint(low=0, high=1024, size=(16, 32))
    # this should fail, as we do not support ndim > 1
    # check in pyg_lib/csrc/ops/index_sort.cpp is not performed
    # TODO: fix this
    sorted_input, indices = pyg_lib.ops.index_sort(input)
