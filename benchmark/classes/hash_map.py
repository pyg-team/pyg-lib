import argparse
import time

import pandas as pd
import torch

from pyg_lib.classes import HashMap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_keys', type=int, default=10_000_000)
    parser.add_argument('--num_queries', type=int, default=1_000_000)
    args = parser.parse_args()

    args.num_queries = min(args.num_queries, args.num_keys)

    num_warmups, num_steps = 50, 100
    if args.device == 'cpu':
        num_warmups, num_steps = num_warmups // 10, num_steps // 10

    max_value = torch.iinfo(torch.long).max

    key1 = torch.randint(0, max_value, (args.num_keys, ), dtype=torch.long,
                         device=args.device)
    query1 = key1[torch.randperm(key1.size(0), device=args.device)]
    query1 = query1[:args.num_queries]

    key2 = torch.randperm(args.num_keys, device=args.device)
    query2 = torch.randperm(args.num_queries, device=args.device)
    query2 = query2[:args.num_queries]

    t_init = t_get = 0
    for i in range(num_warmups + num_steps):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        hash_map = HashMap(key1)
        torch.cuda.synchronize()
        if i >= num_warmups:
            t_init += time.perf_counter() - t_start

        t_start = time.perf_counter()
        hash_map.get(query1)
        torch.cuda.synchronize()
        if i >= num_warmups:
            t_get += time.perf_counter() - t_start

    print(f'HashMap Init: {t_init / num_steps:.4f}s')
    print(f'HashMap  Get: {t_get / num_steps:.4f}s')
    print('=====================')

    t_init = t_get = 0
    for i in range(num_warmups + num_steps):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        hash_map = torch.full((args.num_keys, ), fill_value=-1,
                              dtype=torch.long, device=args.device)
        hash_map[key2] = torch.arange(args.num_keys)
        torch.cuda.synchronize()
        if i >= num_warmups:
            t_init += time.perf_counter() - t_start

        t_start = time.perf_counter()
        hash_map[query2]
        torch.cuda.synchronize()
        if i >= num_warmups:
            t_get += time.perf_counter() - t_start

    print(f' Memory Init: {t_init / num_steps:.4f}s')
    print(f' Memory  Get: {t_get / num_steps:.4f}s')
    print('=====================')

    if key1.is_cpu:
        t_init = t_get = 0
        for i in range(num_warmups + num_steps):
            t_start = time.perf_counter()
            hash_map = pd.CategoricalDtype(categories=key1.numpy(),
                                           ordered=True)
            if i >= num_warmups:
                t_init += time.perf_counter() - t_start

            t_start = time.perf_counter()
            ser = pd.Series(query1.numpy(), dtype=hash_map)
            ser.cat.codes.to_numpy()
            if i >= num_warmups:
                t_get += time.perf_counter() - t_start

    print(f' Pandas Init: {t_init / num_steps:.4f}s')
    print(f' Pandas  Get: {t_get / num_steps:.4f}s')
