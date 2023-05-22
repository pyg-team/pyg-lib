from typing import Tuple, Optional

import torch
from torch import Tensor

NodeType = str
RelType = str
EdgeType = Tuple[str, str, str]


def metis(
    rowptr: Tensor,
    col: Tensor,
    num_partitions: int,
    node_weight: Optional[Tensor] = None,
    edge_weight: Optional[Tensor] = None,
    recursive: bool = False,
) -> Tensor:
    r"""Clusters/partitions a graph into multiple partitions via :obj:`METIS`,
    as motivated by the `"Cluster-GCN: An Efficient Algorithm for Training Deep
    and Large Graph Convolutional Networks"<https://arxiv.org/abs/1905.07953>`_
    paper.

    Args:
        rowptr (torch.Tensor): Compressed source node indices.
        col (torch.Tensor): Target node indices.
        num_partitions (int): The number of partitions.
        node_weight (torch.Tensor, optional): Optional node weights.
            (default: :obj:`None`)
        edge_weight (torch.Tensor, optional): Optional edge weights.
            (default: :obj:`None`)
        recursive (bool, optional): If set to :obj:`True`, will use multilevel
            recursive bisection instead of multilevel k-way partitioning.
            (default: :obj:`False`)

    Returns:
        torch.Tensor: A vector that assings each node to a partition.
    """
    return torch.ops.pyg.metis(rowptr, col, num_partitions, node_weight,
                               edge_weight, recursive)


__all__ = [
    'metis',
]
