from collections.abc import Callable

import torch

_WITH_PT24 = tuple(map(int, torch.__version__.split('.')[:2])) >= (2, 4)

if _WITH_PT24:
    register_fake = torch.library.register_fake
else:

    def register_fake(*args, **kwargs) -> Callable:
        return lambda x: x
