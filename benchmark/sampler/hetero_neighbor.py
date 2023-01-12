import argparse
import ast
import time
from collections import defaultdict
from datetime import datetime
from itertools import product

import pandas as pd
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

# TODO (kgajdamo): Support undirected hetero graphs
# argparser.add_argument('--directed', action='store_true')
argparser.add_argument('--disjoint', action='store_true')
argparser.add_argument('--num_neighbors', type=ast.literal_eval, default=[
    [-1],
    [15, 10, 5],
    [20, 15, 10],
])
# TODO(kgajdamo): Enable sampling with replacement
# argparser.add_argument('--replace', action='store_true')
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--temporal', action='store_true')
argparser.add_argument('--temporal-strategy', choices=['uniform', 'last'],
                       default='uniform')
argparser.add_argument('--write-csv', action='store_true')
argparser.add_argument('--libraries', nargs="*", type=str,
                       default=['pyg-lib', 'torch-sparse'])
args = argparser.parse_args()


@withSeed
@withDataset('ogb', 'mag')
def test_hetero_neighbor(dataset, **kwargs):
    if args.temporal and not args.disjoint:
        raise ValueError(
            "Temporal sampling needs to create disjoint subgraphs")

    colptr_dict, row_dict = dataset
    num_nodes_dict = {k[-1]: v.size(0) - 1 for k, v in colptr_dict.items()}

    if args.temporal:
        # generate random timestamps
        node_time, _ = torch.sort(
            torch.randint(0, 100000, (num_nodes_dict['paper'], )))
        node_time_dict = {'paper': node_time}
    else:
        node_time_dict = None

    if args.shuffle:
        node_perm = torch.randperm(num_nodes_dict['paper'])
    else:
        node_perm = torch.arange(0, num_nodes_dict['paper'])

    data = defaultdict(list)
    for num_neighbors, batch_size in product(args.num_neighbors,
                                             args.batch_sizes):

        print(f'batch_size={batch_size}, num_neighbors={num_neighbors}):')
        data['num_neighbors'].append(num_neighbors)
        data['batch-size'].append(batch_size)

        num_neighbors_dict = {key: num_neighbors for key in colptr_dict.keys()}

        if 'pyg-lib' in args.libraries:
            t = time.perf_counter()
            for seed in tqdm(node_perm.split(batch_size)[:20]):
                seed_dict = {'paper': seed}
                pyg_lib.sampler.hetero_neighbor_sample(
                    colptr_dict,
                    row_dict,
                    seed_dict,
                    num_neighbors_dict,
                    node_time_dict,
                    seed_time_dict=None,
                    csc=True,
                    replace=False,
                    directed=True,
                    disjoint=args.disjoint,
                    temporal_strategy=args.temporal_strategy,
                    return_edge_id=True,
                )
            pyg_lib_duration = time.perf_counter() - t
            data['pyg-lib'].append(round(pyg_lib_duration, 3))
            print(f'     pyg-lib={pyg_lib_duration:.3f} seconds')

        if not args.disjoint:
            if 'torch-sparse' in args.libraries:
                t = time.perf_counter()
                for seed in tqdm(node_perm.split(batch_size)[:20]):
                    node_types = list(num_nodes_dict.keys())
                    edge_types = list(colptr_dict.keys())
                    colptr_dict_sparse = remap_keys(colptr_dict)
                    row_dict_sparse = remap_keys(row_dict)
                    seed_dict = {'paper': seed}
                    num_neighbors_dict_sparse = remap_keys(num_neighbors_dict)
                    num_hops = max([len(v) for v in [num_neighbors]])
                    torch.ops.torch_sparse.hetero_neighbor_sample(
                        node_types,
                        edge_types,
                        colptr_dict_sparse,
                        row_dict_sparse,
                        seed_dict,
                        num_neighbors_dict_sparse,
                        num_hops,
                        False,  # replace
                        True,  # directed
                    )
                torch_sparse_duration = time.perf_counter() - t
                data['torch-sparse'].append(round(torch_sparse_duration, 3))
                print(f'torch-sparse={torch_sparse_duration:.3f} seconds')

            # TODO (kgajdamo): Add dgl hetero sampler.
        print()

    if args.write_csv:
        df = pd.DataFrame(data)
        df.to_csv(f'hetero_neighbor{datetime.now()}.csv', index=False)


if __name__ == '__main__':
    test_hetero_neighbor()
