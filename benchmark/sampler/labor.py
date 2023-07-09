import argparse
import ast
import time
from collections import defaultdict
from datetime import datetime
from itertools import product

import pandas as pd
import torch
from tqdm import tqdm

import pyg_lib
from pyg_lib.testing import withDataset, withSeed

argparser = argparse.ArgumentParser()
argparser.add_argument('--batch-sizes', nargs='+', type=int, default=[
    512,
    1024,
    2048,
    4096,
    8192,
])
argparser.add_argument('--layer-dependencies', nargs='+', type=bool, default=[
    False,
    True,
])
argparser.add_argument('--num_neighbors', type=ast.literal_eval, default=[
    [15, 10, 5],
    [20, 15, 10],
])
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--write-csv', action='store_true')
argparser.add_argument('--libraries', nargs="*", type=str,
                       default=['pyg-lib', 'dgl'])
args = argparser.parse_args()


@withSeed
@withDataset('DIMACS10', 'citationCiteseer')
def test_labor(dataset, **kwargs):
    (rowptr, col), num_nodes = dataset, dataset[0].size(0) - 1
    if 'dgl' in args.libraries:
        import dgl
        dgl_graph = dgl.graph(
            ('csc', (rowptr, col, torch.arange(col.size(0)))))

    if args.shuffle:
        node_perm = torch.randperm(num_nodes)
    else:
        node_perm = torch.arange(num_nodes)

    data = defaultdict(list)
    for num_neighbors, batch_size, layer_dependency in product(
            args.num_neighbors, args.batch_sizes, args.layer_dependencies):

        print(
            f'batch_size={batch_size}, num_neighbors={num_neighbors}), layer_dependency={layer_dependency}:'
        )
        data['num_neighbors'].append(num_neighbors)
        data['batch_size'].append(batch_size)
        data['layer_dependency'].append(layer_dependency)

        if 'pyg-lib' in args.libraries:
            t = time.perf_counter()
            for seed in tqdm(node_perm.split(batch_size)):
                pyg_lib.sampler.labor_sample(
                    rowptr,
                    col,
                    seed,
                    num_neighbors,
                    layer_dependency=layer_dependency,
                    return_edge_id=True,
                )
            pyg_lib_duration = time.perf_counter() - t
            data['pyg-lib'].append(round(pyg_lib_duration, 3))
            print(f'     pyg-lib={pyg_lib_duration:.3f} seconds')

        if 'dgl' in args.libraries:
            import dgl
            dgl_sampler = dgl.dataloading.LaborSampler(
                num_neighbors,
                layer_dependency=layer_dependency,
            )
            dgl_loader = dgl.dataloading.DataLoader(
                dgl_graph,
                node_perm,
                dgl_sampler,
                batch_size=batch_size,
            )
            t = time.perf_counter()
            for _ in tqdm(dgl_loader):
                pass
            dgl_duration = time.perf_counter() - t
            data['dgl'].append(round(dgl_duration, 3))
            print(f'         dgl={dgl_duration:.3f} seconds')
        print()

    if args.write_csv:
        df = pd.DataFrame(data)
        df.to_csv(f'neighbor{datetime.now()}.csv', index=False)


if __name__ == '__main__':
    test_labor()
