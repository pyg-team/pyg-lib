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
// `segment_mean_coo` — Commit 7.
//
// Strategy: reuse the `segment_sum_coo_*` kernels to compute the per-bucket
// sum into `out`, then run a tiny counting kernel that atomically increments
// a per-bucket `int64_t` count buffer, then divide `out` by `count` (clamped
// at 1 to avoid div-by-zero for empty buckets).
//
// Count storage choice: a **separate `int64_t` buffer** of shape
// `[..., out.size(dim)]` (leading dims match `index`, the reduction axis is
// `N = out.size(dim)`). Upstream `pytorch_scatter` (`segment_coo_cuda.cu:
// 199-203, 264-278`) instead reuses the `arg_out` slot typed as the input's
// `scalar_t` and runs the sum kernel with a `nullptr` src to get count-of-1
// behavior. We deviate for two reasons:
//
//   1. Precision: counting in `Half`/`BFloat16` saturates at the dtype's
//      max representable integer (2048 / 256 for `Half`/`BFloat16`).
//   2. Simplicity: a dedicated 1-thread-per-`src`-row counting kernel is
//      trivial and avoids the `HAS_VAL=false` plumbing.
//
// Count tensor is `int64_t`; `atomicAdd(int64_t*, int64_t)` is provided in
// `atomics.cuh`.

// Counting kernel — one thread per `index` entry, atomically increments
// `count[batch_offset + idx]`. `count` has shape `[B, N]` (leading dims of
// `index` with the reduction axis replaced by `N = out.size(dim)`).
__global__ void segment_mean_coo_count_cuda_kernel(
    const at::cuda::detail::TensorInfo<int64_t, int> index_info,
    int64_t* __restrict__ count_data,
    size_t E,
    size_t N) {
  int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int D = index_info.sizes[index_info.dims - 1];

  if (row_idx < E) {
    int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
        row_idx, index_info);
    int64_t idx = index_info.data[offset];
    // `(row_idx / D) * N + idx` picks the count slot inside the per-batch
    // count row, matching the `out_idx` arithmetic in `segment_sum_coo`.
    int count_idx = (row_idx / D) * static_cast<int>(N) + static_cast<int>(idx);
    atomicAdd(count_data + count_idx, static_cast<int64_t>(1));
  }
}

at::Tensor segment_mean_coo_kernel(
    const at::Tensor& src,
    const at::Tensor& index,
    const std::optional<at::Tensor>& optional_out,
    std::optional<int64_t> dim_size) {
  TORCH_CHECK(src.is_cuda(),
              "segment_mean_coo (CUDA): src must be a CUDA tensor");
  TORCH_CHECK(index.is_cuda(),
              "segment_mean_coo (CUDA): index must be a CUDA tensor");
  TORCH_CHECK(src.device() == index.device(),
              "segment_mean_coo (CUDA): src and index must be on the same "
              "device");
  TORCH_CHECK(src.dim() >= index.dim(),
              "segment_mean_coo (CUDA): src.dim() must be >= index.dim() (got ",
              src.dim(), " vs ", index.dim(), ")");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

  // Broadcast `index` up to `src.shape[:index.dim()]` (same rule as the sum
  // kernel; matches upstream `segment_coo_cuda.cu:164-168`).
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
    // Match `segment_sum_coo`'s `out=` handling: caller owns the buffer, no
    // zero-init. The mean semantics with `out=` thus combine the caller's
    // initial value with the new sum before division (mirrors upstream
    // `segment_coo_cuda.cu:175-179, 264-278`).
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "segment_mean_coo (CUDA): out.size(", i,
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
      // Auto-infer from the last (sorted-ascending) index along `dim`.
      auto tail = index_c.select(dim, index_c.size(dim) - 1);
      tail = tail.numel() > 1 ? tail.max() : tail;
      out_sizes[dim] = 1 + tail.cpu().data_ptr<int64_t>()[0];
    }
    // Zero-init so the `atomicAdd` sum pass starts from a clean slate.
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

  // Count buffer: shape `[..., N]` (leading dims of `index`, reduction axis
  // replaced by `N`). Same shape rule upstream uses for the MEAN-reuse
  // `arg_out` (`segment_coo_cuda.cu:200-203`).
  auto count_sizes = index_c.sizes().vec();
  count_sizes[dim] = N;
  auto count = at::zeros(count_sizes, index_c.options());  // `int64_t`

  auto index_info = at::cuda::detail::getTensorInfo<int64_t, int>(index_c);
  const auto stream = at::cuda::getCurrentCUDAStream();

  // -------- Pass 1: sum into `out`. Reuses the `segment_sum_coo_*` kernels.
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "segment_mean_coo_cuda", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();

        if (K == 1) {
          const int T = threads();
          const int B = std::max(1, static_cast<int>((E + T - 1) / T));
          segment_sum_coo_cuda_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, index_info, out_data, E, N);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          // Same `TB` ladder as `segment_sum_coo` — picks the best
          // per-thread inner-loop length for the average run.
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

  // -------- Pass 2: count into `count`. One thread per `index` entry.
  {
    const int T = threads();
    const int B = std::max(1, static_cast<int>((E + T - 1) / T));
    segment_mean_coo_count_cuda_kernel<<<B, T, 0, stream>>>(
        index_info, count.data_ptr<int64_t>(), E, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // -------- Pass 3: divide. Clamp count to >=1 to keep zero-count buckets
  // unchanged (mirrors upstream `arg_out.masked_fill_(arg_out < 1, 1)` at
  // `segment_coo_cuda.cu:269-270`). Then unsqueeze trailing K dims and
  // broadcast-divide.
  count.clamp_min_(1);
  auto count_b = count;
  for (int64_t i = dim + 1; i < out.dim(); ++i) {
    count_b = count_b.unsqueeze(-1);
  }
  if (out.is_floating_point()) {
    out.true_divide_(count_b);
  } else {
    out.div_(count_b, "floor");
  }

  return out;
}

// Per-thread (register-level) `min` for the warp-tree and broadcast inner
// loops. We do **not** use the built-in `<` operator on `at::Half` /
// `at::BFloat16`: when `cuda_fp16.hpp` is included alongside `c10::Half`,
// both an `operator<(__half, __half)` and the built-in `arithmetic <
// arithmetic` participate in overload resolution and nvcc rejects the call as
// ambiguous (same reason `atomicMinHalf` in `atomics.cuh` compares via
// `static_cast<float>`). We dispatch on the dtype: half-precision goes through
// `float`; everything else uses the natural operator.
template <typename scalar_t>
static inline __device__ scalar_t device_min(scalar_t a, scalar_t b) {
  if constexpr (std::is_same_v<scalar_t, at::Half> ||
                std::is_same_v<scalar_t, at::BFloat16>) {
    return static_cast<float>(a) < static_cast<float>(b) ? a : b;
  } else {
    return a < b ? a : b;
  }
}

// Per-thread (register-level) `max`. Symmetric to `device_min` above; see the
// comment there for why we route half-precision through `float`.
template <typename scalar_t>
static inline __device__ scalar_t device_max(scalar_t a, scalar_t b) {
  if constexpr (std::is_same_v<scalar_t, at::Half> ||
                std::is_same_v<scalar_t, at::BFloat16>) {
    return static_cast<float>(a) > static_cast<float>(b) ? a : b;
  } else {
    return a > b ? a : b;
  }
}

// ============================================================================
// `segment_min_coo` — Commit 8.
//
// Two CUDA kernels, launched on the same stream so completion ordering is
// implicit:
//
//   1. Value pass — same structure as `segment_sum_coo`:
//        * K==1: `segment_min_coo_cuda_kernel<scalar_t>` with `SHFL_UP_SYNC`
//          warp-tree reduction over `val` only. The shuffle does **not**
//          propagate `(value, idx)` pairs — argindex is assigned in a
//          separate post-pass. At run boundaries the tail lane writes via
//          `atomicMin(out_data + out_idx, val)` (CAS-loop helper from
//          commit 4 in `atomics.cuh`).
//        * K>1: `segment_min_coo_broadcast_cuda_kernel<scalar_t, TB>` with
//          the same `TB ∈ {4, 8, 16, 32}` ladder as `segment_sum_coo`.
//
//   2. Arg pass — `segment_min_coo_arg_cuda_kernel<scalar_t>` (and a
//      broadcast variant for K>1) re-scans `src` and, where `src[i] ==
//      out[bucket]`, performs `atomicMin(arg_out + bucket, i_within_dim)`
//      over `int64_t`. Initialising `arg_out` to the sentinel
//      `src.size(dim)` makes any valid contributor win the CAS and produces
//      deterministic "first match" semantics (mirrors `scatter_min`'s
//      arg pass).
//
// After both kernels: when we allocated `out` ourselves, we clear empty
// buckets via `out.masked_fill_(arg_out == src.size(dim), 0)` so empty
// buckets produce `0` instead of `numeric_limits::max()`.

// Value-pass K==1 kernel. Mirrors `segment_sum_coo_cuda_kernel` but reduces
// via `min` and writes via `atomicMin`. The shuffle propagates `val` only.
template <typename scalar_t>
__global__ void segment_min_coo_cuda_kernel(
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
      // Warp-tree min reduction: same control flow as the sum kernel, but
      // the per-pair update is `val = min(val, tmp)`. Argindex is NOT
      // propagated through the shuffle — it is reconstructed in the arg
      // pass by matching `src[i] == out[bucket]`.
      tmp = SHFL_UP_SYNC(FULL_MASK, val, i);
      next_idx = SHFL_UP_SYNC(FULL_MASK, idx, i);
      if (lane_idx >= i && row_idx / D == (row_idx - i) / D) {
        if (idx == next_idx) {
          val = device_min(val, tmp);
        }
      }
    }

    // Tail-of-run lane writes via `atomicMin`. Conditions identical to the
    // sum kernel: last lane in warp, outer-batch boundary, or next-row
    // index differs.
    next_idx = SHFL_DOWN_SYNC(FULL_MASK, idx, 1);
    if (lane_idx == WARP_SIZE - 1 || row_idx / D != (row_idx + 1) / D ||
        idx != next_idx) {
      atomicMin(out_data + out_idx, val);
    }
  }
}

// Value-pass K>1 broadcast kernel. Mirrors
// `segment_sum_coo_broadcast_cuda_kernel` but the per-thread accumulator is
// `min` instead of `sum`, and the cross-thread atomic write is `atomicMin`.
template <typename scalar_t, int TB>
__global__ void segment_min_coo_broadcast_cuda_kernel(
    const scalar_t* __restrict__ src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> index_info,
    scalar_t* __restrict__ out_data,
    size_t E,
    size_t K,
    size_t N) {
  int D = index_info.sizes[index_info.dims - 1];
  int E_1 = E / D;
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
      if (idx1 == idx2) {
        scalar_t cand = src_data[K * (dim_start * D + row_start + i) + col_idx];
        val = device_min(val, cand);
      } else {
        atomicMin(out_data + (dim_start * N + idx1) * K + col_idx, val);
        val = src_data[K * (dim_start * D + row_start + i) + col_idx];
      }
      idx1 = idx2;
    }

    atomicMin(out_data + (dim_start * N + idx1) * K + col_idx, val);
  }
}

// Arg-pass K==1 kernel. Re-scans `src` and lowers `arg_out[bucket]` to
// `row_idx % D` (position within the reduction axis) where `src[i] ==
// out[bucket]`. Uses `atomicMin` so the *smallest* `e` wins on ties —
// matches `scatter_min_arg_cuda_kernel`'s deterministic semantics.
template <typename scalar_t>
__global__ void segment_min_coo_arg_cuda_kernel(
    const scalar_t* __restrict__ src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> index_info,
    const scalar_t* __restrict__ out_data,
    int64_t* __restrict__ arg_out_data,
    size_t E,
    size_t N) {
  int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int D = index_info.sizes[index_info.dims - 1];

  if (row_idx < E) {
    int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
        row_idx, index_info);
    int64_t idx = index_info.data[offset];
    int out_idx = (row_idx / D) * static_cast<int>(N) + static_cast<int>(idx);

    // Bit-exact equality. For `at::Half` / `at::BFloat16` we cast through
    // `.x` for the same reason as `scatter_min_arg_cuda_kernel` (see the
    // detailed comment there). The value pass wrote a specific bit pattern;
    // we are checking for that exact pattern in `src`.
    bool match;
    if constexpr (std::is_same_v<scalar_t, at::Half> ||
                  std::is_same_v<scalar_t, at::BFloat16>) {
      match = src_data[row_idx].x == out_data[out_idx].x;
    } else {
      match = src_data[row_idx] == out_data[out_idx];
    }
    if (match) {
      atomicMin(arg_out_data + out_idx, static_cast<int64_t>(row_idx % D));
    }
  }
}

// Arg-pass K>1 broadcast kernel. One thread per `(row, col)` — re-reads
// `out[bucket, col]` and lowers `arg_out[bucket, col]` to `row_idx % D` on
// match.
template <typename scalar_t>
__global__ void segment_min_coo_arg_broadcast_cuda_kernel(
    const scalar_t* __restrict__ src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> index_info,
    const scalar_t* __restrict__ out_data,
    int64_t* __restrict__ arg_out_data,
    size_t E,
    size_t K,
    size_t N) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = thread_idx / static_cast<int>(K);
  int col_idx = thread_idx % static_cast<int>(K);
  int D = index_info.sizes[index_info.dims - 1];

  if (row_idx < static_cast<int>(E) && col_idx < static_cast<int>(K)) {
    int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
        row_idx, index_info);
    int64_t idx = index_info.data[offset];
    int out_idx =
        ((row_idx / D) * static_cast<int>(N) + static_cast<int>(idx)) *
            static_cast<int>(K) +
        col_idx;

    bool match;
    if constexpr (std::is_same_v<scalar_t, at::Half> ||
                  std::is_same_v<scalar_t, at::BFloat16>) {
      match = src_data[thread_idx].x == out_data[out_idx].x;
    } else {
      match = src_data[thread_idx] == out_data[out_idx];
    }
    if (match) {
      atomicMin(arg_out_data + out_idx, static_cast<int64_t>(row_idx % D));
    }
  }
}

std::tuple<at::Tensor, at::Tensor> segment_min_coo_kernel(
    const at::Tensor& src,
    const at::Tensor& index,
    const std::optional<at::Tensor>& optional_out,
    std::optional<int64_t> dim_size) {
  TORCH_CHECK(src.is_cuda(),
              "segment_min_coo (CUDA): src must be a CUDA tensor");
  TORCH_CHECK(index.is_cuda(),
              "segment_min_coo (CUDA): index must be a CUDA tensor");
  TORCH_CHECK(src.device() == index.device(),
              "segment_min_coo (CUDA): src and index must be on the same "
              "device");
  TORCH_CHECK(src.dim() >= index.dim(),
              "segment_min_coo (CUDA): src.dim() must be >= index.dim() (got ",
              src.dim(), " vs ", index.dim(), ")");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

  // Broadcast `index` up to `src.shape[:index.dim()]` (upstream convention;
  // matches the sum/mean kernels above and `segment_coo_cuda.cu:164-168`).
  auto sizes = index.sizes().vec();
  for (int64_t i = 0; i < index.dim(); ++i) {
    sizes[i] = src.size(i);
  }
  auto index_b = index.expand(sizes);

  // COO ops do not take `dim` from Python — it is always `index.dim() - 1`.
  const auto dim = index_b.dim() - 1;

  auto src_c = src.contiguous();
  auto index_c = index_b.contiguous();

  // `E_dim = src.size(dim)` is the sentinel value for `arg_out` (matches
  // `scatter_min`'s sentinel). It is also the per-row D of `src` along the
  // reduction axis.
  const int64_t E_dim = src_c.size(dim);

  at::Tensor out;
  const bool out_was_provided = optional_out.has_value();
  if (out_was_provided) {
    // Caller owns the buffer; no init. The min semantics with `out=` thus
    // combine the caller's initial value with the computed minima via
    // `atomicMin` (matches upstream `segment_coo_cuda.cu:175-179`).
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "segment_min_coo (CUDA): out.size(", i,
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
      auto tail = index_c.select(dim, index_c.size(dim) - 1);
      tail = tail.numel() > 1 ? tail.max() : tail;
      out_sizes[dim] = 1 + tail.cpu().data_ptr<int64_t>()[0];
    }
    // Allocate uninit then fill with per-dtype `max()`. Same rationale as
    // `scatter_min_kernel` (see `scatter_kernel.cu`): `at::full` with a
    // single host `Scalar` can't represent each dtype's max correctly.
    out = at::empty(out_sizes, src_c.options());
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
        "segment_min_coo_cuda_init_out",
        [&] { out.fill_(std::numeric_limits<scalar_t>::max()); });
  }

  // `arg_out` is always allocated and always initialized to the sentinel
  // `E_dim = src.size(dim)`. Even when `out=` is supplied, `arg_out` is
  // fresh (the schema returns it; the caller has no way to pre-allocate).
  auto arg_out = at::full(out.sizes(), E_dim,
                          index_c.options().dtype(at::ScalarType::Long));

  if (index_c.numel() == 0) {
    if (!out_was_provided) {
      // No contributors at all -> every bucket is empty; reset to 0.
      out.zero_();
    }
    return std::make_tuple(out, arg_out);
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
      "segment_min_coo_cuda", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        auto* arg_out_data = arg_out.data_ptr<int64_t>();

        // -------- Value pass.
        if (K == 1) {
          const int T = threads();
          const int B = std::max(1, static_cast<int>((E + T - 1) / T));
          segment_min_coo_cuda_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, index_info, out_data, E, N);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          // Same `TB` ladder as `segment_sum_coo`.
          const dim3 block_dim(32, 8);
          const dim3 grid_cols((K + 31) / 32);
          if (avg_len <= 8.0f) {
            constexpr int TB = 4;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_min_coo_broadcast_cuda_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else if (avg_len <= 16.0f) {
            constexpr int TB = 8;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_min_coo_broadcast_cuda_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else if (avg_len <= 32.0f) {
            constexpr int TB = 16;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_min_coo_broadcast_cuda_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else {
            constexpr int TB = 32;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_min_coo_broadcast_cuda_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        }

        // -------- Arg pass. Same stream -> value-pass writes visible.
        if (K == 1) {
          const int T = threads();
          const int B = std::max(1, static_cast<int>((E + T - 1) / T));
          segment_min_coo_arg_cuda_kernel<scalar_t><<<B, T, 0, stream>>>(
              src_data, index_info, out_data, arg_out_data, E, N);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          const int T = threads();
          const int B = std::max(1, static_cast<int>((E * K + T - 1) / T));
          segment_min_coo_arg_broadcast_cuda_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, index_info, out_data,
                                    arg_out_data, E, K, N);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });

  // Empty-bucket cleanup: where `arg_out == E_dim`, no contributor wrote to
  // that bucket, so `out` still holds `numeric_limits::max()`. Reset to `0`
  // to match the CPU kernel's contract. Skipped when `out=` is supplied —
  // the caller's pre-existing values in empty buckets must be preserved.
  if (!out_was_provided) {
    out.masked_fill_(arg_out == E_dim, 0);
  }

  return std::make_tuple(out, arg_out);
}

// ============================================================================
// `segment_max_coo` — Commit 9.
//
// Symmetric to `segment_min_coo` (commit 8) — same two-pass structure (value
// pass then arg pass on the same stream), same K==1 / K>1 split, same `TB`
// ladder for the broadcast kernel. The only differences from the min path:
//
//   * `out` is initialized to `numeric_limits<scalar_t>::lowest()` (versus
//     `::max()` for min) so the first contributor's `atomicMax` wins.
//   * The warp-tree and broadcast inner loops use `device_max` (`>` rather
//     than `<`) and write via `atomicMax`.
//   * The arg pass still uses `atomicMin(arg_out + bucket, row_idx % D)` —
//     ties resolve to the *first* matching position, matching the CPU
//     kernel's deterministic semantics (strict `>` on the CPU means the
//     first occurrence of the maximum wins; on CUDA we get the same effect
//     by letting the lowest index win the CAS).

// Value-pass K==1 kernel. Mirrors `segment_min_coo_cuda_kernel` with `>` and
// `atomicMax`.
template <typename scalar_t>
__global__ void segment_max_coo_cuda_kernel(
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
      // Warp-tree max reduction. Argindex is reconstructed in the arg pass
      // by matching `src[i] == out[bucket]`.
      tmp = SHFL_UP_SYNC(FULL_MASK, val, i);
      next_idx = SHFL_UP_SYNC(FULL_MASK, idx, i);
      if (lane_idx >= i && row_idx / D == (row_idx - i) / D) {
        if (idx == next_idx) {
          val = device_max(val, tmp);
        }
      }
    }

    // Tail-of-run lane writes via `atomicMax`.
    next_idx = SHFL_DOWN_SYNC(FULL_MASK, idx, 1);
    if (lane_idx == WARP_SIZE - 1 || row_idx / D != (row_idx + 1) / D ||
        idx != next_idx) {
      atomicMax(out_data + out_idx, val);
    }
  }
}

// Value-pass K>1 broadcast kernel. Mirrors
// `segment_min_coo_broadcast_cuda_kernel` with `device_max` / `atomicMax`.
template <typename scalar_t, int TB>
__global__ void segment_max_coo_broadcast_cuda_kernel(
    const scalar_t* __restrict__ src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> index_info,
    scalar_t* __restrict__ out_data,
    size_t E,
    size_t K,
    size_t N) {
  int D = index_info.sizes[index_info.dims - 1];
  int E_1 = E / D;
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
      if (idx1 == idx2) {
        scalar_t cand = src_data[K * (dim_start * D + row_start + i) + col_idx];
        val = device_max(val, cand);
      } else {
        atomicMax(out_data + (dim_start * N + idx1) * K + col_idx, val);
        val = src_data[K * (dim_start * D + row_start + i) + col_idx];
      }
      idx1 = idx2;
    }

    atomicMax(out_data + (dim_start * N + idx1) * K + col_idx, val);
  }
}

// Arg-pass K==1 kernel. Identical to the min variant — the arg pass only
// cares about *matching* values, not the direction of the comparison.
template <typename scalar_t>
__global__ void segment_max_coo_arg_cuda_kernel(
    const scalar_t* __restrict__ src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> index_info,
    const scalar_t* __restrict__ out_data,
    int64_t* __restrict__ arg_out_data,
    size_t E,
    size_t N) {
  int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int D = index_info.sizes[index_info.dims - 1];

  if (row_idx < E) {
    int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
        row_idx, index_info);
    int64_t idx = index_info.data[offset];
    int out_idx = (row_idx / D) * static_cast<int>(N) + static_cast<int>(idx);

    bool match;
    if constexpr (std::is_same_v<scalar_t, at::Half> ||
                  std::is_same_v<scalar_t, at::BFloat16>) {
      match = src_data[row_idx].x == out_data[out_idx].x;
    } else {
      match = src_data[row_idx] == out_data[out_idx];
    }
    if (match) {
      atomicMin(arg_out_data + out_idx, static_cast<int64_t>(row_idx % D));
    }
  }
}

// Arg-pass K>1 broadcast kernel. Identical to the min variant.
template <typename scalar_t>
__global__ void segment_max_coo_arg_broadcast_cuda_kernel(
    const scalar_t* __restrict__ src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> index_info,
    const scalar_t* __restrict__ out_data,
    int64_t* __restrict__ arg_out_data,
    size_t E,
    size_t K,
    size_t N) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = thread_idx / static_cast<int>(K);
  int col_idx = thread_idx % static_cast<int>(K);
  int D = index_info.sizes[index_info.dims - 1];

  if (row_idx < static_cast<int>(E) && col_idx < static_cast<int>(K)) {
    int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
        row_idx, index_info);
    int64_t idx = index_info.data[offset];
    int out_idx =
        ((row_idx / D) * static_cast<int>(N) + static_cast<int>(idx)) *
            static_cast<int>(K) +
        col_idx;

    bool match;
    if constexpr (std::is_same_v<scalar_t, at::Half> ||
                  std::is_same_v<scalar_t, at::BFloat16>) {
      match = src_data[thread_idx].x == out_data[out_idx].x;
    } else {
      match = src_data[thread_idx] == out_data[out_idx];
    }
    if (match) {
      atomicMin(arg_out_data + out_idx, static_cast<int64_t>(row_idx % D));
    }
  }
}

std::tuple<at::Tensor, at::Tensor> segment_max_coo_kernel(
    const at::Tensor& src,
    const at::Tensor& index,
    const std::optional<at::Tensor>& optional_out,
    std::optional<int64_t> dim_size) {
  TORCH_CHECK(src.is_cuda(),
              "segment_max_coo (CUDA): src must be a CUDA tensor");
  TORCH_CHECK(index.is_cuda(),
              "segment_max_coo (CUDA): index must be a CUDA tensor");
  TORCH_CHECK(src.device() == index.device(),
              "segment_max_coo (CUDA): src and index must be on the same "
              "device");
  TORCH_CHECK(src.dim() >= index.dim(),
              "segment_max_coo (CUDA): src.dim() must be >= index.dim() (got ",
              src.dim(), " vs ", index.dim(), ")");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

  // Broadcast `index` up to `src.shape[:index.dim()]` (upstream convention;
  // matches the sum/mean/min kernels above and `segment_coo_cuda.cu:164-168`).
  auto sizes = index.sizes().vec();
  for (int64_t i = 0; i < index.dim(); ++i) {
    sizes[i] = src.size(i);
  }
  auto index_b = index.expand(sizes);

  // COO ops do not take `dim` from Python — it is always `index.dim() - 1`.
  const auto dim = index_b.dim() - 1;

  auto src_c = src.contiguous();
  auto index_c = index_b.contiguous();

  // Sentinel for `arg_out` (matches `scatter_max`'s sentinel).
  const int64_t E_dim = src_c.size(dim);

  at::Tensor out;
  const bool out_was_provided = optional_out.has_value();
  if (out_was_provided) {
    // Caller owns the buffer; no init. With `out=` supplied, the caller's
    // initial values participate in the `atomicMax` reduction.
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "segment_max_coo (CUDA): out.size(", i,
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
      auto tail = index_c.select(dim, index_c.size(dim) - 1);
      tail = tail.numel() > 1 ? tail.max() : tail;
      out_sizes[dim] = 1 + tail.cpu().data_ptr<int64_t>()[0];
    }
    // Allocate uninit then fill with per-dtype `lowest()`. Same rationale as
    // the min kernel — `at::full` with a single host `Scalar` can't represent
    // each dtype's lowest correctly.
    out = at::empty(out_sizes, src_c.options());
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
        "segment_max_coo_cuda_init_out",
        [&] { out.fill_(std::numeric_limits<scalar_t>::lowest()); });
  }

  // `arg_out` is always allocated and always initialized to the sentinel
  // `E_dim = src.size(dim)` (matches the min kernel).
  auto arg_out = at::full(out.sizes(), E_dim,
                          index_c.options().dtype(at::ScalarType::Long));

  if (index_c.numel() == 0) {
    if (!out_was_provided) {
      // No contributors -> every bucket is empty; reset to 0.
      out.zero_();
    }
    return std::make_tuple(out, arg_out);
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
      "segment_max_coo_cuda", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        auto* arg_out_data = arg_out.data_ptr<int64_t>();

        // -------- Value pass.
        if (K == 1) {
          const int T = threads();
          const int B = std::max(1, static_cast<int>((E + T - 1) / T));
          segment_max_coo_cuda_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, index_info, out_data, E, N);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          // Same `TB` ladder as `segment_sum_coo` / `segment_min_coo`.
          const dim3 block_dim(32, 8);
          const dim3 grid_cols((K + 31) / 32);
          if (avg_len <= 8.0f) {
            constexpr int TB = 4;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_max_coo_broadcast_cuda_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else if (avg_len <= 16.0f) {
            constexpr int TB = 8;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_max_coo_broadcast_cuda_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else if (avg_len <= 32.0f) {
            constexpr int TB = 16;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_max_coo_broadcast_cuda_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else {
            constexpr int TB = 32;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_max_coo_broadcast_cuda_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        }

        // -------- Arg pass. Same stream -> value-pass writes visible.
        if (K == 1) {
          const int T = threads();
          const int B = std::max(1, static_cast<int>((E + T - 1) / T));
          segment_max_coo_arg_cuda_kernel<scalar_t><<<B, T, 0, stream>>>(
              src_data, index_info, out_data, arg_out_data, E, N);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          const int T = threads();
          const int B = std::max(1, static_cast<int>((E * K + T - 1) / T));
          segment_max_coo_arg_broadcast_cuda_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, index_info, out_data,
                                    arg_out_data, E, K, N);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });

  // Empty-bucket cleanup: where `arg_out == E_dim`, no contributor wrote to
  // that bucket, so `out` still holds `numeric_limits::lowest()`. Reset to
  // `0` to match the CPU kernel's contract. Skipped when `out=` is
  // supplied — the caller's pre-existing values in empty buckets must be
  // preserved.
  if (!out_was_provided) {
    out.masked_fill_(arg_out == E_dim, 0);
  }

  return std::make_tuple(out, arg_out);
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
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_mean_coo"),
         TORCH_FN(segment_mean_coo_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_min_coo"),
         TORCH_FN(segment_min_coo_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_max_coo"),
         TORCH_FN(segment_max_coo_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::gather_coo"), TORCH_FN(gather_coo_kernel));
}

}  // namespace ops
}  // namespace pyg
