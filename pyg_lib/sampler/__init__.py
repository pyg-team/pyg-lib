from typing import List, Tuple, Optional, Dict

import torch
from torch import Tensor

NodeType = str
RelType = str
EdgeType = Tuple[str, str, str]


def neighbor_sample(
    rowptr: Tensor,
    col: Tensor,
    seed: Tensor,
    num_neighbors: List[int],
    time: Optional[Tensor] = None,
    replace: bool = False,
    directed: bool = True,
    disjoint: bool = False,
    return_edge_id: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    r"""Recursively samples neighbors from all node indices in :obj:`seed`
    in the graph given by :obj:`(rowptr, col)`.

    Args:
        rowptr (torch.Tensor): Compressed source node indices.
        col (torch.Tensor): Target node indices.
        seed (torch.Tensor): The seed node indices.
        num_neighbors (List[int]): The number of neighbors to sample for each
            node in each iteration. If an entry is set to :obj:`-1`, all
            neighbors will be included.
        time (torch.Tensor, optional): Timestamps for the nodes in the graph.
            If set, temporal sampling will be used such that neighbors are
            guaranteed to fulfill temporal constraints, *i.e.* neighbors have
            an earlier timestamp than the seed node.
            Requires :obj:`disjoint=True`. (default: :obj:`None`)
        replace (bool, optional): If set to :obj:`True`, will sample with
            replacement. (default: :obj:`False`)
        directed (bool, optional): If set to :obj:`False`, will include all
            edges between all sampled nodes. (default: :obj:`True`)
        disjoint (bool, optional): If set to :obj:`True` , will create disjoint
            subgraphs for every seed node. (default: :obj:`False`)
        return_edge_id (bool, optional): If set to :obj:`False`, will not
            return the indices of edges of the original graph.
            (default: :obj: `True`)

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]):
        Row indices, col indices of the returned subtree/subgraph, as well as
        original node indices for all nodes sampled.
        In addition, may return the indices of edges of the original graph.
    """
    return torch.ops.pyg.neighbor_sample(rowptr, col, seed, num_neighbors,
                                         time, replace, directed, disjoint,
                                         return_edge_id)


def hetero_neighbor_sample(
    rowptr_dict: Dict[EdgeType, Tensor],
    col_dict: Dict[EdgeType, Tensor],
    seed_dict: Dict[NodeType, Tensor],
    num_neighbors_dict: Dict[EdgeType, List[int]],
    time_dict: Optional[Dict[NodeType, Tensor]] = None,
    replace: bool = False,
    directed: bool = True,
    disjoint: bool = False,
    return_edge_id: bool = True,
) -> Tuple[Dict[EdgeType, Tensor], Dict[EdgeType, Tensor], Dict[
        NodeType, Tensor], Optional[Dict[EdgeType, Tensor]]]:
    r"""Recursively samples neighbors from all node indices in :obj:`seed_dict`
    in the heterogeneous graph given by :obj:`(rowptr_dict, col_dict)`.

    .. note ::
        Similar to :meth:`neighbor_sample`, but expects a dictionary of node
        types (:obj:`str`) and  edge tpyes (:obj:`Tuple[str, str, str]`) for
        each non-boolean argument.

    Args:
        kwargs: Arguments of :meth:`neighbor_sample`.
    """
    src_node_types = {k[0] for k in rowptr_dict.keys()}
    dst_node_types = {k[-1] for k in rowptr_dict.keys()}
    node_types = list(src_node_types | dst_node_types)
    edge_types = list(rowptr_dict.keys())

    TO_REL_TYPE = {key: '__'.join(key) for key in edge_types}
    TO_EDGE_TYPE = {'__'.join(key): key for key in edge_types}

    rowptr_dict = {TO_REL_TYPE[k]: v for k, v in rowptr_dict.items()}
    col_dict = {TO_REL_TYPE[k]: v for k, v in col_dict.items()}
    num_neighbors_dict = {
        TO_REL_TYPE[k]: v
        for k, v in num_neighbors_dict.items()
    }

    out = torch.ops.pyg.hetero_neighbor_sample(
        node_types,
        edge_types,
        rowptr_dict,
        col_dict,
        seed_dict,
        num_neighbors_dict,
        time_dict,
        replace,
        directed,
        disjoint,
        return_edge_id,
    )

    out_row_dict, out_col_dict, out_node_id_dict, out_edge_id_dict = out
    out_row_dict = {TO_EDGE_TYPE[k]: v for k, v in out_row_dict.items()}
    out_col_dict = {TO_EDGE_TYPE[k]: v for k, v in out_col_dict.items()}
    if out_edge_id_dict is not None:
        out_edge_id_dict = {
            TO_EDGE_TYPE[k]: v
            for k, v in out_edge_id_dict.items()
        }

    return out_row_dict, out_col_dict, out_node_id_dict, out_edge_id_dict


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
        In addition, may return the indices of edges of the original graph.
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
    'hetero_neighbor_sample',
    'subgraph',
    'random_walk',
]
