import argparse
import ast
import time

import torch
from tqdm import tqdm

import pyg_lib
from pyg_lib.testing import withDataset, withSeed

try:
    import torch_sparse  # noqa
    baseline_neighbor_sample = torch.ops.torch_sparse.neighbor_sample
except ImportError:
    baseline_neighbor_sample = None

argparser = argparse.ArgumentParser()
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
argparser.add_argument('--directed', action='store_true')
argparser.add_argument('--shuffle', action='store_true')
args = argparser.parse_args()


@withSeed
@withDataset('DIMACS10', 'citationCiteseer')
def test_neighbor(dataset, **kwargs):
    (rowptr, col), num_nodes = dataset, dataset[0].size(0) - 1

    if args.shuffle:
        node_perm = torch.randperm(num_nodes)
    else:
        node_perm = torch.arange(num_nodes)

    for num_neighbors in args.num_neighbors:
        for batch_size in args.batch_sizes:
            print(f'pyg-lib      (batch_size={batch_size}, '
                  f'num_neighbors={num_neighbors}):')
            t = time.perf_counter()
            for seed in tqdm(node_perm.split(batch_size)):
                pyg_lib.sampler.neighbor_sample(
                    rowptr,
                    col,
                    seed,
                    num_neighbors,
                    replace=args.replace,
                    directed=args.directed,
                    disjoint=False,
                    return_edge_id=True,
                )
            print(f'time={time.perf_counter()-t:.3f} seconds')
            print('-------------------------')

            print(f'torch-sparse (batch_size={batch_size}, '
                  f'num_neighbors={num_neighbors}):')
            t = time.perf_counter()
            for seed in tqdm(node_perm.split(batch_size)):
                torch.ops.torch_sparse.neighbor_sample(
                    rowptr,
                    col,
                    seed,
                    num_neighbors,
                    args.replace,
                    args.directed,
                )
            print(f'time={time.perf_counter()-t:.3f} seconds')
            print('-------------------------')


if __name__ == '__main__':
    test_neighbor()
