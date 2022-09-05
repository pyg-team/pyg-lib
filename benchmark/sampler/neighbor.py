import argparse
import ast
import time

import torch

try:
    import torch_sparse  # noqa
    baseline_neighbor_sample = torch.ops.torch_sparse.neighbor_sample
except ImportError:
    baseline_neighbor_sample = None
from tqdm import tqdm

import pyg_lib
from pyg_lib.testing import withDataset, withSeed

argparser = argparse.ArgumentParser('Neighbor Sampler benchmark')
argparser.add_argument('--batch-sizes', nargs='+',
                       default=[512, 1024, 2048, 4096, 8192], type=int)
argparser.add_argument('--num_neighbors', default=[[-1], [15, 10, 5],
                                                   [20, 15, 10]],
                       type=ast.literal_eval)
argparser.add_argument('--replace', action='store_true')
argparser.add_argument('--directed', action='store_true')
argparser.add_argument('--shuffle', action='store_true')

args = argparser.parse_args()


@withSeed
@withDataset('DIMACS10', 'citationCiteseer')
def test_neighbor(dataset, **kwargs):
    (rowptr, col), num_nodes = dataset, dataset[0].size(0) - 1

    for num_neighbors in args.num_neighbors:
        for batch_size in args.batch_sizes:
            # pyg-lib neighbor sampler
            start = time.perf_counter()
            nodes_ids = torch.randperm(
                num_nodes) if args.shuffle else torch.arange(0, num_nodes)
            for seed in tqdm(nodes_ids.split(batch_size)):
                pyg_lib.sampler.neighbor_sample(rowptr, col, seed,
                                                num_neighbors,
                                                replace=args.replace,
                                                directed=args.directed,
                                                disjoint=False,
                                                return_edge_id=True)
            stop = time.perf_counter()
            print('-------------------------')
            print('pyg-lib neighbor sample')
            print(f'Batch size={batch_size}, '
                  f'Num_neighbors={num_neighbors}, '
                  f'Time={stop-start:.3f} seconds\n')

            # pytorch-sparse neighbor sampler
            start = time.perf_counter()
            for seed in tqdm(nodes_ids.split(batch_size)):
                torch.ops.torch_sparse.neighbor_sample(rowptr, col, seed,
                                                       num_neighbors,
                                                       args.replace,
                                                       args.directed)
            stop = time.perf_counter()
            print('-------------------------')
            print('pytorch_sparse neighbor sample')
            print(f'Batch size={batch_size}, '
                  f'Num_neighbors={num_neighbors}, '
                  f'Time={stop-start:.3f} seconds\n')


if __name__ == '__main__':
    test_neighbor()
