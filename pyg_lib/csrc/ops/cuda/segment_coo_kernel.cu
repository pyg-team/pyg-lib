#include "../segment_coo.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include "atomics.cuh"
#include "shuffle.cuh"

namespace pyg {
namespace ops {

namespace {

// Convention 13: WARP_SIZE is fixed at 32 on every NVIDIA architecture pyg-lib
// targets (sm_60+). Already defined by `scatter_kernel.cu` in its own TU; we
// redefine guarded here so the two TUs stay independent.
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Full-warp mask for `__shfl_*_sync` — every lane participates.
#define FULL_MASK 0xffffffff

// Convention 12: dynamic block sizing. Replicates the inline `threads()` /
// `blocks()` pattern from `scatter_kernel.cu` (which itself is a copy of
// `pyg_lib/csrc/sampler/cuda/random_walk_kernel.cu`). Each new kernel file in
// `pyg_lib/csrc/ops/cuda/` carries its own copy until a shared helper lands.
int threads() {
  const auto props = at::cuda::getCurrentDeviceProperties();
  return std::min(props->maxThreadsPerBlock, 1024);
}

int blocks(int numel) {
  const auto props = at::cuda::getCurrentDeviceProperties();
  const auto blocks_per_sm = props->maxThreadsPerMultiProcessor / 256;
  const auto max_blocks = props->multiProcessorCount * blocks_per_sm;
  const auto max_threads = threads();
  return std::max(
      1, std::min(max_blocks, (numel + max_threads - 1) / max_threads));
}

// ============================================================================
// `segment_sum_coo` — Commit 6.
//
// Two CUDA kernel variants:
//
//   1. `segment_sum_coo_cuda_kernel` (K==1, no trailing feature dims). One
//   thread
//      per `src` entry. `SHFL_UP_SYNC` performs a warp-level tree reduction
//      across equal indices within a single warp. Lanes that hold the tail of
//      a run (next-lane index differs or warp boundary) atomically write the
//      accumulated value into `out[index]`. Port of upstream
//      `pytorch_scatter/csrc/cuda/segment_coo_cuda.cu:14-53` — see also the
//      design note in `PLAN_PYTORCH_SCATTER.md:363-366`.
//
//   2. `segment_sum_coo_broadcast_cuda_kernel<scalar_t, TB>` (K>1). Each thread
//      processes a single column and `TB` consecutive index entries. The
//      host-side dispatcher chooses `TB ∈ {4, 8, 16, 32}` based on the
//      average run length `avg_len = index.size(dim) / dim_size`. Port of
//      upstream `segment_coo_cuda.cu:76-127, 225-248`.

// K==1 kernel — one thread per `src` entry, warp-reduce equal-index runs.
//
// `index_info` is `at::cuda::detail::TensorInfo<int64_t, int>` so we can
// resolve a possibly-strided `index` view through `IndexToOffset` without
// materializing a contiguous copy. The dispatcher passes the linear-strided
// `.contiguous()` index, but the TensorInfo handle is the cheapest way to
// reuse upstream's offset arithmetic verbatim.
template <typename scalar_t>
__global__ void segment_sum_coo_cuda_kernel(
    const scalar_t* __restrict__ src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> index_info,
    scalar_t* __restrict__ out_data,
    size_t E,
    size_t N) {
  int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lane_idx = row_idx & (WARP_SIZE - 1);
  int D = index_info.sizes[index_info.dims - 1];

  if (row_idx < E) {
    int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
        row_idx, index_info);
    int64_t idx = index_info.data[offset];
    int64_t next_idx;
    int out_idx = (row_idx / D) * N + idx;

    scalar_t val = src_data[row_idx];
    scalar_t tmp;

#pragma unroll
    for (int i = 1; i < WARP_SIZE; i *= 2) {
      // Parallel reduction inside a single warp. `SHFL_UP_SYNC` pulls the
      // value/index from lane `lane_idx - i`; if that lane belongs to the
      // same outer-batch and shares the same `idx`, we fold it into `val`.
      tmp = SHFL_UP_SYNC(FULL_MASK, val, i);
      next_idx = SHFL_UP_SYNC(FULL_MASK, idx, i);
      if (lane_idx >= i && row_idx / D == (row_idx - i) / D) {
        // Upstream asserts `idx >= next_idx` (sorted-index UB contract).
        if (idx == next_idx) {
          val = val + tmp;
        }
      }
    }

    // The lane that holds the *tail* of a run writes the accumulated value
    // to `out`. Conditions for being a tail:
    //   * last lane in the warp (`lane_idx == WARP_SIZE - 1`), or
    //   * boundary in the outer-batch dimension (`row_idx / D != (row_idx +
    //     1) / D`), or
    //   * next-row's index differs from this row's.
    next_idx = SHFL_DOWN_SYNC(FULL_MASK, idx, 1);
    if (lane_idx == WARP_SIZE - 1 || row_idx / D != (row_idx + 1) / D ||
        idx != next_idx) {
      atomicAdd(out_data + out_idx, val);
    }
  }
}

// K>1 broadcast kernel. Each thread owns one column `(col_idx)` and `TB`
// consecutive index entries starting at `(dim_start, row_start)`. We sweep
// the `TB` entries in registers, accumulating equal-index runs and writing
// out to `out` via `atomicAdd` whenever the run ends (next index differs or
// we hit the `TB` boundary). Port of upstream
// `segment_coo_broadcast_kernel`.
//
// `TB` is a compile-time literal (4, 8, 16, or 32) so the `#pragma unroll`
// over the inner loop bites and the per-thread register pressure stays
// predictable.
template <typename scalar_t, int TB>
__global__ void segment_sum_coo_broadcast_cuda_kernel(
    const scalar_t* __restrict__ src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> index_info,
    scalar_t* __restrict__ out_data,
    size_t E,
    size_t K,
    size_t N) {
  int D = index_info.sizes[index_info.dims - 1];
  int E_1 = E / D;
  // Round `D-1` up to a multiple of `TB`. The thread that lands on
  // `dim_start * D + row_start + i` for `i ∈ [0, TB)` only does work while
  // `row_start + i < D`.
  int E_2 = (D - 1) + TB - ((D - 1) % TB);

  int row_idx = blockIdx.x * blockDim.y + threadIdx.y;
  int col_idx = blockIdx.y * blockDim.x + threadIdx.x;

  int dim_start = (row_idx * TB) / E_2;
  int row_start = (row_idx * TB) % E_2;

  if (dim_start < E_1 && col_idx < K) {
    int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
        dim_start * D + row_start, index_info);
    int64_t idx1 = index_info.data[offset];
    int64_t idx2;

    scalar_t val = src_data[K * (dim_start * D + row_start) + col_idx];

#pragma unroll
    for (int i = 1; i < TB; ++i) {
      if (row_start + i >= D)
        break;

      idx2 =
          index_info.data[offset + i * index_info.strides[index_info.dims - 1]];
      // Upstream asserts `idx1 <= idx2` (sorted-index UB contract).
      if (idx1 == idx2) {
        val = val + src_data[K * (dim_start * D + row_start + i) + col_idx];
      } else {
        atomicAdd(out_data + (dim_start * N + idx1) * K + col_idx, val);
        val = src_data[K * (dim_start * D + row_start + i) + col_idx];
      }
      idx1 = idx2;
    }

    atomicAdd(out_data + (dim_start * N + idx1) * K + col_idx, val);
  }
}

at::Tensor segment_sum_coo_kernel(const at::Tensor& src,
                                  const at::Tensor& index,
                                  const std::optional<at::Tensor>& optional_out,
                                  std::optional<int64_t> dim_size) {
  TORCH_CHECK(src.is_cuda(),
              "segment_sum_coo (CUDA): src must be a CUDA tensor");
  TORCH_CHECK(index.is_cuda(),
              "segment_sum_coo (CUDA): index must be a CUDA tensor");
  TORCH_CHECK(src.device() == index.device(),
              "segment_sum_coo (CUDA): src and index must be on the same "
              "device");
  TORCH_CHECK(src.dim() >= index.dim(),
              "segment_sum_coo (CUDA): src.dim() must be >= index.dim() (got ",
              src.dim(), " vs ", index.dim(), ")");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

  // Broadcast `index` up to `src.shape[:index.dim()]` (upstream convention;
  // matches `segment_coo_cpu.cpp` and `segment_coo_cuda.cu:164-168`).
  auto sizes = index.sizes().vec();
  for (int64_t i = 0; i < index.dim(); ++i) {
    sizes[i] = src.size(i);
  }
  auto index_b = index.expand(sizes);

  // COO ops do not take `dim` from Python — it is always `index.dim() - 1`.
  const auto dim = index_b.dim() - 1;

  auto src_c = src.contiguous();
  auto index_c = index_b.contiguous();

  at::Tensor out;
  const bool out_was_provided = optional_out.has_value();
  if (out_was_provided) {
    // Accumulate-into contract: caller owns the buffer, no zero-init.
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "segment_sum_coo (CUDA): out.size(", i,
                    ") must match src.size(", i, ")");
      }
    }
  } else {
    auto out_sizes = src_c.sizes().vec();
    if (dim_size.has_value()) {
      out_sizes[dim] = dim_size.value();
    } else if (index_c.numel() == 0) {
      out_sizes[dim] = 0;
    } else {
      // Auto-infer from the last index along `dim` — sorted-index contract
      // means this is the maximum index. Mirrors upstream
      // `segment_coo_cuda.cu:187-189`.
      auto tail = index_c.select(dim, index_c.size(dim) - 1);
      tail = tail.numel() > 1 ? tail.max() : tail;
      out_sizes[dim] = 1 + tail.cpu().data_ptr<int64_t>()[0];
    }
    // Zero-init so the `atomicAdd` accumulation starts from a clean slate.
    out = at::zeros(out_sizes, src_c.options());
  }

  if (index_c.numel() == 0) {
    return out;
  }

  const auto E = index_c.numel();
  const auto E_2 = index_c.size(dim);
  const auto E_1 = E / E_2;
  const auto K = src_c.numel() / E;
  const auto N = out.size(dim);
  const float avg_len = static_cast<float>(E_2) / static_cast<float>(N);

  auto index_info = at::cuda::detail::getTensorInfo<int64_t, int>(index_c);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "segment_sum_coo_cuda", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();

        if (K == 1) {
          // One thread per `src` entry; warp-reduce equal-index runs in
          // shuffle, then a single `atomicAdd` per run's tail. The launch
          // grid uses the dynamic `threads()` / `blocks(E)` helpers; the
          // shuffle reduction depends only on warp lanes (warp-relative
          // indices via `row_idx & 31`), not block geometry, so any block
          // size that is a multiple of WARP_SIZE works. `threads()` always
          // returns a multiple of 32 on every supported arch.
          const int T = threads();
          const int B = std::max(1, static_cast<int>((E + T - 1) / T));
          segment_sum_coo_cuda_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, index_info, out_data, E, N);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          // Broadcast path. Pick `TB ∈ {4, 8, 16, 32}` from `avg_len` — the
          // larger the average run, the more entries each thread should
          // sweep before writing out. Mirrors upstream
          // `segment_coo_cuda.cu:225-248`.
          //
          // The grid geometry is `(rows_blocks, cols_blocks)` with block
          // `(32, 8)`: 32 threads along `K` (one warp per column tile) and
          // 8 threads along the row axis. We launch enough row blocks to
          // cover `E_1 * ceil(E_2 / TB)` row-positions in groups of 8.
          const dim3 block_dim(32, 8);
          const dim3 grid_cols((K + 31) / 32);
          if (avg_len <= 8.0f) {
            constexpr int TB = 4;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_sum_coo_broadcast_cuda_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else if (avg_len <= 16.0f) {
            constexpr int TB = 8;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_sum_coo_broadcast_cuda_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else if (avg_len <= 32.0f) {
            constexpr int TB = 16;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_sum_coo_broadcast_cuda_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else {
            constexpr int TB = 32;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_sum_coo_broadcast_cuda_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        }
      });

  return out;
}

// ============================================================================
// `gather_coo` — Commit 6.
//
// Trivial gather: `out[i] = src[index[i]]` along `dim = index.dim() - 1`,
// with broadcasting over the trailing feature dims. Two kernel variants:
//
//   * K==1: one thread per output element, straight `__ldg`-load gather.
//   * K>1: one thread per `(row, col)` pair, same load pattern.
//
// `out=` contract: overwrite (not accumulate). When `out=` is supplied we
// write into the caller's buffer; otherwise we allocate `at::empty(...)`.
// No zero-init in either case — every output element is written exactly once.

template <typename scalar_t>
__global__ void gather_coo_cuda_kernel(
    const scalar_t* __restrict__ src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> index_info,
    scalar_t* __restrict__ out_data,
    size_t E,
    size_t N) {
  int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (row_idx < E) {
    int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
        row_idx, index_info);
    int row = static_cast<int>(index_info.data[offset]);

    // `(row_idx / D) * N` picks the outer-batch slice of `src` and `row`
    // is the index within that slice. `D = index.size(dim)`.
    offset =
        (row_idx / index_info.sizes[index_info.dims - 1]) * static_cast<int>(N);
    out_data[row_idx] = src_data[offset + row];
  }
}

template <typename scalar_t>
__global__ void gather_coo_broadcast_cuda_kernel(
    const scalar_t* __restrict__ src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> index_info,
    scalar_t* __restrict__ out_data,
    size_t E,
    size_t K,
    size_t N) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = thread_idx / static_cast<int>(K);
  int col_idx = thread_idx % static_cast<int>(K);

  if (thread_idx < static_cast<int>(E * K)) {
    int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
        row_idx, index_info);
    int row = static_cast<int>(index_info.data[offset]);

    offset = (row_idx / index_info.sizes[index_info.dims - 1]) *
             static_cast<int>(N) * static_cast<int>(K);
    out_data[thread_idx] =
        src_data[offset + static_cast<int>(K) * row + col_idx];
  }
}

at::Tensor gather_coo_kernel(const at::Tensor& src,
                             const at::Tensor& index,
                             const std::optional<at::Tensor>& optional_out) {
  TORCH_CHECK(src.is_cuda(), "gather_coo (CUDA): src must be a CUDA tensor");
  TORCH_CHECK(index.is_cuda(),
              "gather_coo (CUDA): index must be a CUDA tensor");
  TORCH_CHECK(src.device() == index.device(),
              "gather_coo (CUDA): src and index must be on the same device");
  TORCH_CHECK(src.dim() >= index.dim(),
              "gather_coo (CUDA): src.dim() must be >= index.dim() (got ",
              src.dim(), " vs ", index.dim(), ")");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

  // Broadcast `index`'s leading dims up to `src.shape[:index.dim() - 1]`.
  // Upstream uses `index.dim() - 1` as the broadcast bound (not `index.dim()`)
  // because `index.size(dim)` is the output's *reduction-axis* extent, not a
  // batch dim. Matches `segment_coo_cuda.cu:337-340`.
  auto sizes = index.sizes().vec();
  for (int64_t i = 0; i < index.dim() - 1; ++i) {
    sizes[i] = src.size(i);
  }
  auto index_b = index.expand(sizes);

  const auto dim = index_b.dim() - 1;

  auto src_c = src.contiguous();
  auto index_c = index_b.contiguous();

  at::Tensor out;
  if (optional_out.has_value()) {
    // Overwrite contract: every output element is written exactly once.
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < src_c.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "gather_coo (CUDA): out.size(", i, ") must match src.size(",
                    i, ")");
      }
    }
    TORCH_CHECK(index_c.size(dim) == out.size(dim),
                "gather_coo (CUDA): out.size(", dim, ") must match index.size(",
                dim, ")");
  } else {
    auto out_sizes = src_c.sizes().vec();
    out_sizes[dim] = index_c.size(dim);
    out = at::empty(out_sizes, src_c.options());
  }

  if (index_c.numel() == 0) {
    return out;
  }

  const auto E = index_c.numel();
  const auto K = out.numel() / E;
  const auto N = src_c.size(dim);

  auto index_info = at::cuda::detail::getTensorInfo<int64_t, int>(index_c);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "gather_coo_cuda", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();

        if (K == 1) {
          const int T = threads();
          const int B = blocks(static_cast<int>(E));
          gather_coo_cuda_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, index_info, out_data, E, N);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          const int T = threads();
          const int B = blocks(static_cast<int>(E * K));
          gather_coo_broadcast_cuda_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, index_info, out_data, E, K, N);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_sum_coo"),
         TORCH_FN(segment_sum_coo_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::gather_coo"), TORCH_FN(gather_coo_kernel));
}

}  // namespace ops
}  // namespace pyg
