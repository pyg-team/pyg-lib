import pytest
import torch

import pyg_lib

DEVICES = [torch.device('cpu')]
if torch.cuda.is_available():
    DEVICES.append(torch.device('cuda'))


@pytest.mark.skipif(
    torch.__version__.startswith('2.4.0'),
    reason="https://github.com/pytorch/pytorch/issues/130619",
)
@pytest.mark.parametrize('device', [
    pytest.param('cpu'),
    pytest.param(
        'cuda', marks=pytest.mark.skipif(
            not torch.cuda.is_available(),
            reason="CUDA is not available",
        )),
])
def test_index_sort(device):
    inputs = torch.randperm(100_000, device=device)
    ref_sorted_input, ref_indices = torch.sort(inputs, stable=True)
    sorted_input, indices = pyg_lib.ops.index_sort(inputs)
    assert torch.all(ref_sorted_input == sorted_input)
    assert torch.all(ref_indices == indices)


def test_index_sort_invalid():
    inputs = torch.randint(low=0, high=1024, size=(16, 32))
    with pytest.raises(RuntimeError):
        sorted_input, indices = pyg_lib.ops.index_sort(inputs)
