import os
import os.path as osp
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from pyg_lib import get_home_dir

# Decorators ##################################################################


def withSeed(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        torch.manual_seed(12345)
        func(*args, **kwargs)

    return wrapper


def onlyCUDA(func: Callable) -> Callable:
    import pytest
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )(func)


def onlyTriton(func: Callable) -> Callable:
    import pytest

    return pytest.mark.skipif(
        find_spec('triton') is None,
        reason="'triton' not installed",
    )(func)


def withCUDA(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        func(*args, device=torch.device('cpu'), **kwargs)
        if torch.cuda.is_available():
            func(*args, device=torch.device('cuda:0'), **kwargs)

    return wrapper


def withDataset(group: str, name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if group == 'ogb' and name == 'mag':
                dataset = get_ogb_mag_hetero_sparse_matrix(
                    dtype=kwargs.get('dtype', torch.long),
                    device=kwargs.get('device', None),
                )
            else:
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


def get_ogb_mag_hetero_sparse_matrix(
    dtype: torch.dtype = torch.long,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Returns a heterogeneous graph :obj:`(colptr_dict, row_dict)`
    from the `OGB <https://ogb.stanford.edu/>`_ benchmark suite.

    Args:
        dtype (torch.dtype, optional): The desired data type of returned
            tensors. (default: :obj:`torch.long`)
        device (torch.device, optional): the desired device of returned
            tensors. (default: :obj:`None`)

    Returns:
        (Dict[Tuple[str, str, str], torch.Tensor],
        Dict[Tuple[str, str, str], torch.Tensor], int, List,
        List[Tuple[str, str, str]]): Compressed source node indices and target
        node indices of the hetero sparse matrix, number of paper nodes,
        all node types and all edge types.
    """
    import torch_geometric.transforms as T
    from torch_geometric.datasets import OGB_MAG

    path = osp.join(get_home_dir(), 'ogb-mag')
    transform = T.Compose([T.ToUndirected(), T.ToSparseTensor()])
    data = OGB_MAG(path, pre_transform=transform)[0]

    colptr_dict, row_dict = {}, {}
    for edge_type in data.edge_types:
        colptr, row, _ = data[edge_type].adj_t.csr()
        colptr_dict[edge_type] = colptr.to(device, dtype)
        row_dict[edge_type] = row.to(device, dtype)

    return colptr_dict, row_dict


def to_edge_index(rowptr: Tensor, col: Tensor) -> Tensor:
    row = torch.arange(rowptr.size(0) - 1, dtype=col.dtype, device=col.device)
    row = row.repeat_interleave(rowptr[1:] - rowptr[:-1])
    return torch.stack([row, col], dim=0)


def remap_keys(mapping: Dict[Tuple[str, str, str], Any]) -> Dict[str, Any]:
    return {'__'.join(k): v for k, v in mapping.items()}
