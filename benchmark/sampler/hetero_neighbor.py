import argparse
import ast
import os.path as osp
import time

import torch
import torch_geometric.transforms as T
import torch_sparse  # noqa
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader.utils import get_input_nodes
from torch_geometric.sampler.utils import remap_keys, to_hetero_csc
from tqdm import tqdm

import pyg_lib
from pyg_lib.testing import withSeed

argparser = argparse.ArgumentParser('Hetero neighbor sample benchmark')
argparser.add_argument('--batch-sizes', nargs='+', type=int, default=[
    512,
    1024,
    2048,
    4096,
    8192,
])
argparser.add_argument('--num_neighbors', type=ast.literal_eval, default=[
    [-1],
    [15, 10, 5],
    [20, 15, 10],
])
argparser.add_argument('--replace', action='store_true')
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--do_train', action='store_true')
# TODO (kgajdamo): Support undirected hetero graphs
# argparser.add_argument('--directed', action='store_true')
# TODO (kgajdamo): Enable CSR
# argparser.add_argument('--csc', action='store_true')

args = argparser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/OGB')
transform = T.ToUndirected(merge=True)
dataset = OGB_MAG(path, preprocess='metapath2vec', transform=transform)

data = dataset[0].to(device, 'x', 'y')

train_input_nodes = ('paper', data['paper'].train_mask)
val_input_nodes = ('paper', data['paper'].val_mask)

if args.do_train:
    node_type, input_nodes = get_input_nodes(data, train_input_nodes)
else:
    node_type, input_nodes = get_input_nodes(data, val_input_nodes)


@withSeed
def test_hetero_neighbor():
    node_types, edge_types = data.metadata()
    colptr, row, _ = to_hetero_csc(data, device=device, share_memory=False,
                                   is_sorted=False)

    # Conversions to/from C++ string type:
    to_rel_type = {key: '__'.join(key) for key in edge_types}
    colptr_sparse = remap_keys(colptr, to_rel_type)
    row_sparse = remap_keys(row, to_rel_type)

    if args.shuffle:
        node_perm = input_nodes[torch.randperm(input_nodes.size()[0])]
    else:
        node_perm = input_nodes

    for num_neighbors in args.num_neighbors:
        num_hops = max([0] + [len(v) for v in [num_neighbors]])
        num_neighbors_dict = {key: num_neighbors for key in edge_types}

        for batch_size in args.batch_sizes:
            print(f'batch_size={batch_size}, num_neighbors={num_neighbors}):')
            t = time.perf_counter()
            for seed in tqdm(node_perm.split(batch_size)):
                seed_dict = {'paper': seed}
                pyg_lib.sampler.hetero_neighbor_sample(
                    colptr,
                    row,
                    seed_dict,
                    num_neighbors_dict,
                    None,
                    True,  # csc
                    args.replace,
                    True,  # directed
                )
            pyg_lib_duration = time.perf_counter() - t

            num_neighbors_sparse = remap_keys(num_neighbors_dict, to_rel_type)
            t = time.perf_counter()
            for seed in tqdm(node_perm.split(batch_size)):
                seed_dict = {'paper': seed}
                torch.ops.torch_sparse.hetero_neighbor_sample(
                    node_types,
                    edge_types,
                    colptr_sparse,
                    row_sparse,
                    seed_dict,
                    num_neighbors_sparse,
                    num_hops,
                    args.replace,
                    True,  # directed
                )
            torch_sparse_duration = time.perf_counter() - t

            # TODO (kgajdamo): Add dgl hetero sampler?

            print(f'     pyg-lib={pyg_lib_duration:.3f} seconds')
            print(f'torch-sparse={torch_sparse_duration:.3f} seconds')
            print()


if __name__ == '__main__':
    test_hetero_neighbor()
