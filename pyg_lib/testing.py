import os
import os.path as osp
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from pyg_lib import get_home_dir

# Decorators ##################################################################


def withSeed(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        torch.manual_seed(12345)
        func(*args, **kwargs)

    return wrapper


def withCUDA(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        func(*args, device=torch.device('cpu'), **kwargs)
        if torch.cuda.is_available():
            func(*args, device=torch.device('cuda:0'), **kwargs)

    return wrapper


def withDataset(group: str, name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            dataset = get_sparse_matrix(
                group,
                name,
                dtype=kwargs.get('dtype', torch.long),
                device=kwargs.get('device', None),
            )

            func(*args, dataset=dataset, **kwargs)

        return wrapper

    return decorator


# Helper functions ############################################################


def get_sparse_matrix(
    group: str,
    name: str,
    dtype: torch.dtype = torch.long,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Returns a sparse matrix :obj:`(rowptr, col)` from the
    `Suite Sparse Matrix Collection <https://sparse.tamu.edu>`_.

    Args:
        group (string): The group of the sparse matrix.
        name (string): The name of the sparse matrix.
        dtype (torch.dtype, optional): The desired data type of returned
            tensors. (default: :obj:`torch.long`)
        device (torch.device, optional): the desired device of returned
            tensors. (default: :obj:`None`)

    Returns:
        (torch.Tensor, torch.Tensor): Compressed source node indices and target
        node indices of the sparse matrix.
    """
    path = osp.join(get_home_dir(), f'{name}.mat')
    if not osp.exists(path):
        os.makedirs(get_home_dir(), exist_ok=True)

        import urllib
        url = f'https://sparse.tamu.edu/mat/{group}/{name}.mat'
        print(f'Downloading {url}...', end='')
        data = urllib.request.urlopen(url)
        with open(path, 'wb') as f:
            f.write(data.read())
        print(' Done!')

    from scipy.io import loadmat
    mat = loadmat(path)['Problem'][0][0][2].tocsr()

    rowptr = torch.from_numpy(mat.indptr).to(device, dtype)
    col = torch.from_numpy(mat.indices).to(device, dtype)

    return rowptr, col


def to_edge_index(rowptr: Tensor, col: Tensor) -> Tensor:
    row = torch.arange(rowptr.size(0) - 1, dtype=col.dtype, device=col.device)
    row = row.repeat_interleave(rowptr[1:] - rowptr[:-1])
    return torch.stack([row, col], dim=0)
