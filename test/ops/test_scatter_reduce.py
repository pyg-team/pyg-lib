import torch
import triton
import triton.language as tl

from pyg_lib.ops import fused_scatter_reduce
from pyg_lib.testing import onlyCUDA, onlyTriton


# todo tune block size
# todo group indices
# @triton.autotune(configs=[
#     triton.Config(meta={'BLOCK_SIZE': 256}),
#     triton.Config(meta={'BLOCK_SIZE': 512}),
#     triton.Config(meta={'BLOCK_SIZE': 1024}),
# ], key=['num_feats'])
@triton.jit
def scatter_add_kernel(inputs_ptr, index_ptr, out_ptr, M, N, **meta):
    block_size_M = meta['BLOCK_SIZE_M']
    tile_size_M = meta['TILE_SIZE_M']
    block_size_N = meta['BLOCK_SIZE_N']

    # block_start_M = tl.program_id(axis=0) * block_size_M * tile_size_M
    # block_start_N = tl.program_id(axis=1) * block_size_N

    # N_offset = block_start_N + tl.arange(0, block_size_N)
    # N_mask = N_offset < N

    # for i in range(0, tile_size_M):
    #     M_offset = block_start_M + tl.arange(0, block_size_M) * tile_size_M + i
    #     inputs_offset = N * M_offset[:, None] + N_offset[None, :]
    #     inputs_mask = (M_offset < M)[:, None] * N_mask[None, :]
    #     inputs = tl.load(inputs_ptr + inputs_offset, mask=inputs_mask, other=0)

    # pid = tl.program_id(axis=0)
    # block_start = pid * meta['BLOCK_SIZE']

    # offsets = block_start + tl.arange(0, meta['BLOCK_SIZE'])
    # mask = offsets < numel

    # inputs = tl.load(inputs_ptr + offsets, mask=mask)

    # index_offsets = offsets // num_feats
    # index = tl.load(index_ptr + index_offsets, mask=mask)

    # out_offsets = num_feats * index + (offsets % num_feats)
    # tl.atomic_add(out_ptr + out_offsets, inputs, mask=mask)


def scatter_add(inputs, index, dim_size: int):
    out = inputs.new_zeros(dim_size, inputs.size(-1))

    grid = lambda meta: (
        triton.cdiv(inputs.size(0), meta['BLOCK_SIZE_M'] * meta['TILE_SIZE_M']
                    ),
        triton.cdiv(inputs.size(1), meta['BLOCK_SIZE_N']),
    )

    scatter_add_kernel[grid](inputs, index, out, inputs.size(0),
                             inputs.size(1), TILE_SIZE_M=8, BLOCK_SIZE_M=8,
                             BLOCK_SIZE_N=32)

    return out


@onlyCUDA
@onlyTriton
def test_fused_scatter_reduce():
    x = torch.randn(5, 4, device='cuda')
    index = torch.tensor([0, 1, 0, 1, 0], device='cuda')

    out = fused_scatter_reduce(x, index, dim_size=2,
                               reduce_list=['sum', 'mean'])

    assert out.size() == (2, 8)
    assert torch.allclose(out[0, 0:4], x[index == 0].sum(dim=0))
    assert torch.allclose(out[1, 0:4], x[index == 1].sum(dim=0))
    assert torch.allclose(out[0, 4:8], x[index == 0].mean(dim=0))
    assert torch.allclose(out[1, 4:8], x[index == 1].mean(dim=0))


if __name__ == '__main__':  # Benchmarking
    import time

    import torch_scatter

    dim_size = 1000
    x = torch.randn(50000, 128, device='cuda')
    index = torch.randint(dim_size, (x.size(0), ), device='cuda')
    index = index.sort()[0]

    num_warmups = 1000
    num_steps = 10000

    out1 = scatter_add(x, index, dim_size=dim_size)
    out2 = torch_scatter.scatter_add(x, index, dim_size=dim_size, dim=0)
    print(torch.allclose(out1, out2, atol=1e-3))

    # for i in range(num_warmups + num_steps):
    #     if i == num_warmups:
    #         torch.cuda.synchronize()
    #         t = time.perf_counter()
    #     out_fused = fused_scatter_reduce(x, index, dim_size=dim_size,
    #                                      reduce_list=['sum', 'mean'])
    # torch.cuda.synchronize()
    # t = time.perf_counter() - t
    # print(f'  Fused implementation: {t:.4f} seconds')

    for i in range(num_warmups + num_steps):
        if i == num_warmups:
            torch.cuda.synchronize()
            t = time.perf_counter()
        out1 = torch_scatter.scatter_add(x, index, dim_size=dim_size, dim=0)
        out2 = torch_scatter.scatter_add(x, index, dim_size=dim_size, dim=0)
        # out = torch.cat([out1, out2], dim=-1)
    torch.cuda.synchronize()
    t = time.perf_counter() - t
    print(f'Vanilla implementation: {t:.4f} seconds')

    for i in range(num_warmups + num_steps):
        if i == num_warmups:
            torch.cuda.synchronize()
            t = time.perf_counter()
        out1 = scatter_add(x, index, dim_size=dim_size)
    torch.cuda.synchronize()
    t = time.perf_counter() - t
    print(f'single implementation: {t:.4f} seconds')

    for i in range(num_warmups + num_steps):
        if i == num_warmups:
            torch.cuda.synchronize()
            t = time.perf_counter()
        out1 = torch_scatter.segment_add_coo(x, index, dim_size=dim_size)
    torch.cuda.synchronize()
    t = time.perf_counter() - t
    print(f'sorted implementation: {t:.4f} seconds')

    # assert torch.allclose(out_fused, out_vanilla, atol=1e-5)
