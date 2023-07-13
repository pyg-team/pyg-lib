import argparse
import ast
import time
from collections import defaultdict
from datetime import datetime
from itertools import accumulate, product

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
argparser.add_argument('--num_neighbors', type=ast.literal_eval, default=[
    [10, 10, 10],
    [20, 15, 10],
])
argparser.add_argument('--with-prob', nargs='+', type=bool,
                       default=[True, False])
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

    node_perm = torch.randperm(num_nodes)
    probs = torch.rand(col.shape[0])

    data = defaultdict(list)
    for num_neighbors, batch_size, with_prob in product(
            args.num_neighbors, args.batch_sizes, args.with_prob):

        print(
            f'batch_size={batch_size}, num_neighbors={num_neighbors}), with_prob={with_prob}'
        )
        data['num_neighbors'].append(num_neighbors)
        data['batch_size'].append(batch_size)
        data['with_prob'].append(with_prob)

        if 'pyg-lib' in args.libraries:
            t = time.perf_counter()
            num_edges = [0 for _ in num_neighbors]
            num_nodes = [0] + [0 for _ in num_neighbors]
            num_batches = 0
            for seed in tqdm(node_perm.split(batch_size)):
                r = pyg_lib.sampler.labor_sample(
                    rowptr,
                    col,
                    seed,
                    num_neighbors,
                    probs=(probs if with_prob else None),
                    return_edge_id=True,
                )
                for i, _ in enumerate(num_neighbors):
                    num_edges[i] += r[5][i]
                    num_nodes[i] += r[4][i]
                num_nodes[-1] += r[4][-1]
                num_batches += 1
            pyg_lib_duration = time.perf_counter() - t
            avg_edges = [e // num_batches for e in accumulate(num_edges)]
            avg_nodes = [n // num_batches for n in accumulate(num_nodes)]
            data['pyg-lib'].append(round(pyg_lib_duration, 3))
            data['pyg-lib-avg-edges'].append(avg_edges)
            data['pyg-lib-avg-nodes'].append(avg_nodes)
            print(f'     pyg-lib={pyg_lib_duration:.3f} seconds, '
                  f'avg_edges: {avg_edges}, avg_nodes: {avg_nodes}')

        if 'dgl' in args.libraries:
            import dgl
            dgl_sampler = dgl.dataloading.LaborSampler(
                list(reversed(num_neighbors)),
                layer_dependency=True,
                prob=(probs if with_prob else None),
            )
            dgl_loader = dgl.dataloading.DataLoader(
                dgl_graph,
                node_perm,
                dgl_sampler,
                batch_size=batch_size,
            )
            t = time.perf_counter()
            num_edges = [0 for _ in num_neighbors]
            num_nodes = [0] + [0 for _ in num_neighbors]
            num_batches = 0
            for r in tqdm(dgl_loader):
                num_nodes[0] += r[1].shape[0]
                for i, block in enumerate(r[2]):
                    num_edges[-i - 1] += block.num_edges()
                    num_nodes[-i - 1] += block.num_src_nodes()
                num_batches += 1
            dgl_duration = time.perf_counter() - t
            avg_edges = [e // num_batches for e in num_edges]
            avg_nodes = [n // num_batches for n in num_nodes]
            data['dgl'].append(round(dgl_duration, 3))
            data['dgl-avg-edges'].append(avg_edges)
            data['dgl-avg-nodes'].append(avg_nodes)
            print(f'         dgl={dgl_duration:.3f} seconds, '
                  f'avg_edges: {avg_edges}, avg_nodes: {avg_nodes}')
        print()

    if args.write_csv:
        df = pd.DataFrame(data)
        df.to_csv(f'labor{datetime.now()}.csv', index=False)


if __name__ == '__main__':
    test_labor()
