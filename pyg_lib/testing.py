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


def withDataset(group: str, name: str,
                return_csc: Optional[bool] = False) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            dataset = get_sparse_matrix(
                group,
                name,
                dtype=kwargs.get('dtype', torch.long),
                device=kwargs.get('device', None),
                return_csc=return_csc,
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
    return_csc: Optional[bool] = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    r"""Returns a sparse matrix :obj:`(rowptr, col)` from the
    `Suite Sparse Matrix Collection <https://sparse.tamu.edu>`_.
    In addition, may return a sparse matrix in CSC format,
    then output will be :obj:`(rowptr, col, colptr, row)`.

    Args:
        group (string): The group of the sparse matrix.
        name (string): The name of the sparse matrix.
        dtype (torch.dtype, optional): The desired data type of returned
            tensors. (default: :obj:`torch.long`)
        device (torch.device, optional): the desired device of returned
            tensors. (default: :obj:`None`)
        return_csc (bool, optional): If set to :obj:`True`, will additionaly
            return a sparse matrix in CSC format. (default: :obj:`False`)

    Returns:
        (torch.Tensor, torch.Tensor, Optional[torch.Tensor],
        Optional[torch.Tensor]): Compressed source node indices and target node
        indices of the sparse matrix. In addition, may return a sparse matrix
        in CSC format.
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
    csr_mat = loadmat(path)['Problem'][0][0][2].tocsr()

    rowptr = torch.from_numpy(csr_mat.indptr).to(device, dtype)
    col = torch.from_numpy(csr_mat.indices).to(device, dtype)

    if return_csc:
        csc_mat = loadmat(path)['Problem'][0][0][2].tocsc()

        colptr = torch.from_numpy(csc_mat.indptr).to(device, dtype)
        row = torch.from_numpy(csc_mat.indices).to(device, dtype)

        return rowptr, col, colptr, row

    return rowptr, col


def to_edge_index(rowptr: Tensor, col: Tensor) -> Tensor:
    row = torch.arange(rowptr.size(0) - 1, dtype=col.dtype, device=col.device)
    row = row.repeat_interleave(rowptr[1:] - rowptr[:-1])
    return torch.stack([row, col], dim=0)
