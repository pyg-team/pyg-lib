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
    seed_time: Optional[Tensor] = None,
    csc: bool = False,
    replace: bool = False,
    directed: bool = True,
    disjoint: bool = False,
    temporal_strategy: str = 'uniform',
    return_edge_id: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], List[int], List[int]]:
    r"""Recursively samples neighbors from all node indices in :obj:`seed`
    in the graph given by :obj:`(rowptr, col)`.

    .. note::

        For temporal sampling, the :obj:`col` vector needs to be sorted
        according to :obj:`time` within individual neighborhoods since we use
        binary search to find neighbors that fulfill temporal constraints.

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
            an earlier or equal timestamp than the seed node.
            If used, the :obj:`col` vector needs to be sorted according to time
            within individual neighborhoods. Requires :obj:`disjoint=True`.
            (default: :obj:`None`)
        seed_time (torch.Tensor, optional): Optional values to override the
            timestamp for seed nodes. If not set, will use timestamps in
            :obj:`time` as default for seed nodes. (default: :obj:`None`)
        csc (bool, optional): If set to :obj:`True`, assumes that the graph is
            given in CSC format :obj:`(colptr, row)`. (default: :obj:`False`)
        replace (bool, optional): If set to :obj:`True`, will sample with
            replacement. (default: :obj:`False`)
        directed (bool, optional): If set to :obj:`False`, will include all
            edges between all sampled nodes. (default: :obj:`True`)
        disjoint (bool, optional): If set to :obj:`True` , will create disjoint
            subgraphs for every seed node. (default: :obj:`False`)
        temporal_strategy (string, optional): The sampling strategy when using
            temporal sampling (:obj:`"uniform"`, :obj:`"last"`).
            (default: :obj:`"uniform"`)
        return_edge_id (bool, optional): If set to :obj:`False`, will not
            return the indices of edges of the original graph.
            (default: :obj: `True`)

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor],
        List[int], List[int]):
        Row indices, col indices of the returned subtree/subgraph, as well as
        original node indices for all nodes sampled.
        In addition, may return the indices of edges of the original graph.
        Lastly, returns information about the sampled amount of nodes and edges
        per hop.
    """
    return torch.ops.pyg.neighbor_sample(rowptr, col, seed, num_neighbors,
                                         time, seed_time, csc, replace,
                                         directed, disjoint, temporal_strategy,
                                         return_edge_id)


def hetero_neighbor_sample(
    rowptr_dict: Dict[EdgeType, Tensor],
    col_dict: Dict[EdgeType, Tensor],
    seed_dict: Dict[NodeType, Tensor],
    num_neighbors_dict: Dict[EdgeType, List[int]],
    time_dict: Optional[Dict[NodeType, Tensor]] = None,
    seed_time_dict: Optional[Dict[NodeType, Tensor]] = None,
    csc: bool = False,
    replace: bool = False,
    directed: bool = True,
    disjoint: bool = False,
    temporal_strategy: str = 'uniform',
    return_edge_id: bool = True,
) -> Tuple[Dict[EdgeType, Tensor], Dict[EdgeType, Tensor], Dict[
        NodeType, Tensor], Optional[Dict[EdgeType, Tensor]], Dict[
            NodeType, List[int]], Dict[NodeType, List[int]]]:
    r"""Recursively samples neighbors from all node indices in :obj:`seed_dict`
    in the heterogeneous graph given by :obj:`(rowptr_dict, col_dict)`.

    .. note ::
        Similar to :meth:`neighbor_sample`, but expects a dictionary of node
        types (:obj:`str`) and  edge types (:obj:`Tuple[str, str, str]`) for
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
        seed_time_dict,
        csc,
        replace,
        directed,
        disjoint,
        temporal_strategy,
        return_edge_id,
    )

    (row_dict, col_dict, node_id_dict, edge_id_dict, num_nodes_per_hop_dict,
     num_edges_per_hop_dict) = out

    row_dict = {TO_EDGE_TYPE[k]: v for k, v in row_dict.items()}
    col_dict = {TO_EDGE_TYPE[k]: v for k, v in col_dict.items()}

    if edge_id_dict is not None:
        edge_id_dict = {TO_EDGE_TYPE[k]: v for k, v in edge_id_dict.items()}

    num_edges_per_hop_dict = {
        TO_EDGE_TYPE[k]: v
        for k, v in num_edges_per_hop_dict.items()
    }

    return (row_dict, col_dict, node_id_dict, edge_id_dict,
            num_nodes_per_hop_dict, num_edges_per_hop_dict)


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
