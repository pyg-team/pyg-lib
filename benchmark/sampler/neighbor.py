import argparse
import time

import torch
# uncomment below to enable torch.ops.torch_sparse.neighbor_sample
# import torch_sparse
from tqdm import tqdm

import pyg_lib
from pyg_lib.testing import withDataset, withSeed

argparser = argparse.ArgumentParser('Neighbor Sampler benchmark')
argparser.add_argument('--batch-sizes', nargs='+',
                       default=[512, 1024, 2048, 4096, 8192], type=int)
argparser.add_argument('--num_neighbors', default=[[-1], [15, 10, 5],
                                                   [20, 15, 10]], type=int)
argparser.add_argument('--replace', default=False, type=bool)
argparser.add_argument('--directed', default=True, type=bool)

args = argparser.parse_args()


@withSeed
@withDataset('DIMACS10', 'citationCiteseer', True)
def test_neighbor(dataset, **kwargs):
    (rowptr, col, colptr, row), num_nodes = dataset, dataset[0].size(0) - 1

    for num_neighbors in args.num_neighbors:
        for batch_size in args.batch_sizes:
            # pyg-lib neighbor sampler
            start = time.perf_counter()
            for node in tqdm(range(num_nodes - batch_size)):
                seed = torch.arange(node, node + batch_size)
                pyg_lib.sampler.neighbor_sample(rowptr, col, seed,
                                                num_neighbors, args.replace,
                                                args.directed, disjoint=False,
                                                return_edge_id=True)
            stop = time.perf_counter()
            print('-------------------------')
            print('pyg-lib neighbor sample')
            print(f'Batch size={batch_size}, '
                  f'Num_neighbors={num_neighbors}, '
                  f'Inference time={stop-start:.3f} seconds\n')

            # pytorch-sparse neighbor sampler
            start = time.perf_counter()
            for node in tqdm(range(num_nodes - batch_size)):
                seed = torch.arange(node, node + batch_size)
                torch.ops.torch_sparse.neighbor_sample(colptr, row, seed,
                                                       num_neighbors,
                                                       args.replace,
                                                       args.directed)
            stop = time.perf_counter()
            print('-------------------------')
            print('pytorch_sparse neighbor sample')
            print(f'Batch size={batch_size}, '
                  f'Num_neighbors={num_neighbors}, '
                  f'Inference time={stop-start:.3f} seconds\n')


if __name__ == '__main__':
    test_neighbor()