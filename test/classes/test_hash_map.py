import os.path as osp

import pytest
import torch
from torch import Tensor

from pyg_lib.testing import withCUDA

INT_TO_DTPYE = {
    2: torch.short,
    3: torch.int,
    4: torch.long,
}


@withCUDA
@pytest.mark.parametrize('dtype', [torch.short, torch.int, torch.long])
def test_hash_map(dtype, device):
    key = torch.tensor([0, 10, 30, 20], device=device, dtype=dtype)
    query = torch.tensor([30, 10, 20, 40], device=device, dtype=dtype)

    if key.is_cpu:
        HashMap = torch.classes.pyg.CPUHashMap
        hash_map = HashMap(key, 0)
    elif key.is_cuda:
        HashMap = torch.classes.pyg.CUDAHashMap
        hash_map = HashMap(key, 0.5)
    else:
        raise NotImplementedError(f"Unsupported device '{device}'")

    assert hash_map.size() == 4
    assert INT_TO_DTPYE[hash_map.dtype()] == dtype
    assert hash_map.device() == device
    assert hash_map.keys().equal(key)
    assert hash_map.keys().dtype == dtype
    expected = torch.tensor([2, 1, 3, -1], device=device)
    assert hash_map.get(query).equal(expected)
    assert hash_map.get(query).dtype == torch.long

    if key.is_cpu:  # Test parallel hash map:
        hash_map = HashMap(key, 16)
        assert hash_map.keys().equal(key)
        assert hash_map.keys().dtype == dtype
        assert hash_map.get(query).equal(expected)
        assert hash_map.get(query).dtype == torch.long


class Foo(torch.nn.Module):
    def __init__(self, key: Tensor):
        super().__init__()
        if key.is_cpu:
            HashMap = torch.classes.pyg.CPUHashMap
            self.map = HashMap(key, 0)
        elif key.is_cuda:
            HashMap = torch.classes.pyg.CUDAHashMap
            self.map = HashMap(key, 0.5)


@withCUDA
def test_serialization(device, tmp_path):
    key = torch.tensor([0, 10, 30, 20], device=device)
    scripted_foo = torch.jit.script(Foo(key))

    path = osp.join(tmp_path, 'foo.pt')
    scripted_foo.save(path)
    loaded = torch.jit.load(path)

    assert loaded.map.keys().equal(key)
