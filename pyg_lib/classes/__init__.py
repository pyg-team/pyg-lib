import torch
from torch import Tensor


class HashMap:
    def __init__(self, key: Tensor) -> Tensor:
        self._map = torch.classes.pyg.CPUHashMap(key)

    def get(self, query: Tensor) -> Tensor:
        return self._map.get(query)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


__all__ = [
    'HashMap',
]
