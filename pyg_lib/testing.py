import os
import os.path as osp
from typing import Tuple

import torch
from torch import Tensor

from pyg_lib import get_home_dir


def get_sparse_matrix(
    group: str,
    name: str,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> Tuple[Tensor, Tensor]:
    r"""Returns a sparse matrix :obj:`(rowptr, col)` from the
    `Suite Sparse Matrix Collection <https://sparse.tamu.edu>`_.

    Args:
        group (string): The group of the sparse matrix.
        name (string): The name of the sparse matrix.

    Returns:
        (torch.Tensor, torch.Tensor): Compressed source node indices and target
        node indices of the sparse matrix.
    """
    dtype = dtype if dtype is not None else torch.long

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
