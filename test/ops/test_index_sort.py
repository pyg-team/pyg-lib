import torch

import pyg_lib
import pytest

DEVICES = [torch.device('cpu')]
if torch.cuda.is_available():
    DEVICES.append(torch.device('cuda'))

torch.manual_seed(1234)


@pytest.mark.parametrize('device', DEVICES)
def test_index_sort(device):
    input = torch.randint(low=0, high=1024, size=(1000000, ), device=device)
    ref_sorted_input, ref_indices = torch.sort(input, stable=True)
    sorted_input, indices = pyg_lib.ops.index_sort(input)
    assert torch.all(ref_sorted_input == sorted_input)
    assert torch.all(ref_indices == indices)


@pytest.mark.parametrize('device', DEVICES)
def test_index_sort_negative(device):
    input = torch.randint(low=0, high=1024, size=(16, 32), device=device)
    with pytest.raises(RuntimeError):
        sorted_input, indices = pyg_lib.ops.index_sort(input)
