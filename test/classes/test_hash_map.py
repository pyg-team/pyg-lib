import pytest
import torch

from pyg_lib.testing import withCUDA


@withCUDA
@pytest.mark.parametrize('load_factor', [0.5, 0.25])
@pytest.mark.parametrize('dtype', [torch.short, torch.int, torch.long])
def test_hash_map(load_factor, dtype, device):
    key = torch.tensor([0, 10, 30, 20], device=device, dtype=dtype)
    query = torch.tensor([30, 10, 20, 40], device=device, dtype=dtype)

    if key.is_cpu:
        HashMap = torch.classes.pyg.CPUHashMap
        hash_map = HashMap(key, 0, load_factor)
    elif key.is_cuda:
        HashMap = torch.classes.pyg.CUDAHashMap
        hash_map = HashMap(key, load_factor)
    else:
        raise NotImplementedError(f"Unsupported device '{device}'")

    assert hash_map.keys().equal(key)
    assert hash_map.keys().equal(key)
    expected = torch.tensor([2, 1, 3, -1], device=device)
    assert hash_map.get(query).equal(expected)
    assert hash_map.get(query).dtype == torch.long

    if key.is_cpu:
        hash_map = HashMap(key, 16, load_factor)
        assert hash_map.keys().dtype == dtype
        assert hash_map.keys().equal(key)
        assert hash_map.get(query).equal(expected)
        assert hash_map.get(query).dtype == torch.long
