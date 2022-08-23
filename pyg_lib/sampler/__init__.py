from typing import Tuple, Optional

import torch
from torch import Tensor


def neighbor_sample(
    rowptr: Tensor, col: Tensor, seed: Tensor, num_neighbors: list[int],
    replace: bool = False, directed: bool = True, isolated: bool = True,
    return_edge_id: bool = True
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    r"""Recursively samples neighbors from all nodes indices in :obj:`seed`
    in the graph given by :obj:`(rowptr, col)`.

    Args:
        rowptr (torch.Tensor): Compressed source node indices.
        col (torch.Tensor): Target node indices.
        seed (torch.Tensor):
        num_neighbors (list[int]): The number of neighbors to sample for each
            node in each iteration. In heterogeneous graphs, may also take in a
            dictionary denoting the amount of neighbors to sample for each
            individual edge type. If an entry is set to :obj:`-1`, all neighbors
            will be included.
        replace (bool, optional): If set to :obj:`True`, will sample with
            replacement. (default: :obj:`False`)
        directed (bool, optional): If set to :obj:`False`, will include all
            edges between all sampled nodes. (default: :obj:`True`)
        isolated (bool, optional): #TODO(kgajdamo): add description
        return_edge_id (bool, optional): If set to :obj:`False`, will not
            return the indices of edges of the original graph.
            (default: :obj: `True`)

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]):
        #TODO(kgajdamo): add description
    """
    return torch.ops.pyg.neighbor_sample(rowptr, col, seed, num_neighbors,
                                         replace, directed, isolated,
                                         return_edge_id)


def subgraph(
    rowptr: Tensor,
    col: Tensor,
    nodes: Tensor,
    return_edge_id: bool = True,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    r"""Returns the induced subgraph of the graph given by
    :obj:`(rowptr, col)`, containing only the nodes in :obj:`nodes`.

    Args:
        rowptr (torch.Tensor): Compressed source node indices.
        col (torch.Tensor): Target node indices.
        nodes (torch.Tensor): Node indices of the induced subgraph.
        return_edge_id (bool, optional): If set to :obj:`False`, will not
            return the indices of edges of the original graph contained in the
            induced subgraph. (default: :obj:`True`)

    Returns:
        (torch.Tensor, torch.Tensor, Optional[torch.Tensor]): Compressed source
        node indices and target node indices of the induced subgraph.
        In addition, may return the indices of edges of the original graph
        contained in the induced subgraph.
    """
    return torch.ops.pyg.subgraph(rowptr, col, nodes, return_edge_id)


def random_walk(rowptr: Tensor, col: Tensor, seed: Tensor, walk_length: int,
                p: float = 1.0, q: float = 1.0) -> Tensor:
    r"""Samples random walks of length :obj:`walk_length` from all node
    indices in :obj:`seed` in the graph given by :obj:`(rowptr, col)`, as
    described in the `"node2vec: Scalable Feature Learning for Networks"
    <https://arxiv.org/abs/1607.00653>`_ paper.

    Args:
        rowptr (torch.Tensor): Compressed source node indices.
        col (torch.Tensor): Target node indices.
        seed (torch.Tensor): Seed node indices from where random walks start.
        walk_length (int): The walk length of a random walk.
        p (float, optional): Likelihood of immediately revisiting a node in the
            walk. (default: :obj:`1.0`)
        q (float, optional): Control parameter to interpolate between
            breadth-first strategy and depth-first strategy.
            (default: :obj:`1.0`)

    Returns:
        torch.Tensor: A tensor of shape :obj:`[seed.size(0), walk_length + 1]`,
        holding the nodes indices of each walk for each seed node.
    """
    return torch.ops.pyg.random_walk(rowptr, col, seed, walk_length, p, q)


__all__ = [
    'neighbor_sample',
    'subgraph',
    'random_walk',
]
