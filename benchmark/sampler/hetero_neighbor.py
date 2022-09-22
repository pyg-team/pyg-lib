import argparse
import ast
import time

import torch
import torch_sparse  # noqa
from tqdm import tqdm

import pyg_lib
from pyg_lib.testing import remap_keys, withDataset, withSeed

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
# TODO(kgajdamo): Enable sampling with replacement
# argparser.add_argument('--replace', action='store_true')
argparser.add_argument('--shuffle', action='store_true')
# TODO (kgajdamo): Support undirected hetero graphs
# argparser.add_argument('--directed', action='store_true')

args = argparser.parse_args()


@withSeed
@withDataset('ogbn', 'mag')
def test_hetero_neighbor(dataset, **kwargs):
    rowptr_dict, col_dict, num_nodes, node_types, edge_types = dataset

    # Conversions to/from C++ string type:
    to_rel_type = {key: '__'.join(key) for key in edge_types}
    rowptr_sparse = remap_keys(rowptr_dict, to_rel_type)
    col_sparse = remap_keys(col_dict, to_rel_type)

    if args.shuffle:
        node_perm = torch.randperm(num_nodes)
    else:
        node_perm = torch.arange(0, num_nodes)

    for num_neighbors in args.num_neighbors:
        num_hops = max([0] + [len(v) for v in [num_neighbors]])
        num_neighbors_dict = {key: num_neighbors for key in edge_types}

        for batch_size in args.batch_sizes:
            print(f'batch_size={batch_size}, num_neighbors={num_neighbors}):')
            t = time.perf_counter()
            for seed in tqdm(node_perm.split(batch_size)):
                seed_dict = {'paper': seed}
                pyg_lib.sampler.hetero_neighbor_sample(
                    rowptr_dict,
                    col_dict,
                    seed_dict,
                    num_neighbors_dict,
                    None,
                    False,  # csc
                    False,
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
                    rowptr_sparse,
                    col_sparse,
                    seed_dict,
                    num_neighbors_sparse,
                    num_hops,
                    False,
                    True,  # directed
                )
            torch_sparse_duration = time.perf_counter() - t

            # TODO (kgajdamo): Add dgl hetero sampler?

            print(f'     pyg-lib={pyg_lib_duration:.3f} seconds')
            print(f'torch-sparse={torch_sparse_duration:.3f} seconds')
            print()


if __name__ == '__main__':
    test_hetero_neighbor()
