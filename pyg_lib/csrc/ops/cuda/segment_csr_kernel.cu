#include "../segment_csr.h"

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include "shuffle.cuh"

namespace pyg {
namespace ops {

namespace {

// Convention 13: WARP_SIZE is fixed at 32 on every NVIDIA architecture pyg-lib
// targets (sm_60+). Already defined in `segment_coo_kernel.cu` /
// `scatter_kernel.cu` in their own TUs; we redefine guarded here so the TUs
// stay independent.
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Full-warp mask for `__shfl_*_sync` — every lane participates.
#ifndef FULL_MASK
#define FULL_MASK 0xffffffff
#endif

// Strict less-than predicate used by the min reductions below. Half-precision
// routes through `float` to avoid an ambiguous overload between
// `c10::Half::operator<` and the built-in `arithmetic <` (nvcc rejects the
// direct comparison; same reason `device_min` in `segment_coo_kernel.cu`
// uses a `static_cast<float>` route).
template <typename scalar_t>
static inline __device__ bool device_lt(scalar_t a, scalar_t b) {
  if constexpr (std::is_same_v<scalar_t, at::Half> ||
                std::is_same_v<scalar_t, at::BFloat16>) {
    return static_cast<float>(a) < static_cast<float>(b);
  } else {
    return a < b;
  }
}

// Strict greater-than predicate used by the max reductions below. Same
// half-precision routing as `device_lt`.
template <typename scalar_t>
static inline __device__ bool device_gt(scalar_t a, scalar_t b) {
  if constexpr (std::is_same_v<scalar_t, at::Half> ||
                std::is_same_v<scalar_t, at::BFloat16>) {
    return static_cast<float>(a) > static_cast<float>(b);
  } else {
    return a > b;
  }
}

// Convention 12: dynamic block sizing — replicates the inline `threads()` /
// `blocks()` pattern used in `segment_coo_kernel.cu` / `scatter_kernel.cu`.
// Each `.cu` file in `pyg_lib/csrc/ops/cuda/` carries its own copy until a
// shared helper lands.
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

// Strided-offset helper specialised for `indptr` row pointers.
//
// `at::cuda::detail::IndexToOffset` walks all dims of a `TensorInfo` and is
// suitable for COO `index` (where the last dim's size matches the number of
// reduced rows). For CSR `indptr` we need the **same arithmetic with the last
// dim's effective size reduced by 1** — `indptr` has `rows + 1` entries per
// batch (one trailing sentinel), but we only index `[0, rows)` from the
// kernel. Port of upstream `pytorch_scatter/csrc/cuda/index_info.cuh:7-19`.
struct IndexPtrToOffset {
  static __host__ __device__ __forceinline__ int get(
      int idx,
      const at::cuda::detail::TensorInfo<int64_t, int>& info) {
    int offset = idx % (info.sizes[info.dims - 1] - 1);
    offset *= info.strides[info.dims - 1];
    idx /= info.sizes[info.dims - 1] - 1;
    for (int i = info.dims - 2; i >= 0; --i) {
      offset += (idx % info.sizes[i]) * info.strides[i];
      idx /= info.sizes[i];
    }
    return offset;
  }
};

// ============================================================================
// `segment_sum_csr` — Commit 10.
//
// Two CUDA kernel variants:
//
//   1. `segment_sum_csr_cuda_kernel` (K==1, no trailing feature dims). One
//      thread per row; sequential inner loop summing
//      `src[indptr[r] : indptr[r+1]]` into `out[r]`. Port of upstream
//      `pytorch_scatter/csrc/cuda/segment_csr_cuda.cu:15-60` with `TB == 1`
//      — TB==1 means single thread per row, so the warp-tree shuffle path is
//      dead code and is omitted entirely. The `SHFL_DOWN_SYNC` reduction only
//      fires for the min/max ops (commits 12/13) where TB > 1.
//
//   2. `segment_sum_csr_broadcast_cuda_kernel` (K>1, trailing feature dims).
//      One thread per `(row, col)` pair; each thread sequentially walks the
//      row's source range. Port of upstream `segment_csr_cuda.cu:62-95`.
//      Per the upstream comment at line 70, shared-memory reuse was tried
//      and discarded — the syncthreads overhead exceeded the benefit, so we
//      keep the same "no shared memory" design here.
//
// Accumulation is in `at::opmath_type<scalar_t>` registers; the final cast
// back to `scalar_t` matches upstream behaviour for `Half`/`BFloat16`.
//
// `out=` contract: when the caller supplies `optional_out`, we
// **accumulate** into it (no zero-init). Matches the upstream `segment_csr`
// contract and is what makes `GatherCSR::backward` efficient (`at::zeros` +
// `segment_sum_csr(out=grad_in)` deposits the gradient in-place).

// K==1 kernel — one thread per row.
template <typename scalar_t>
__global__ void segment_sum_csr_cuda_kernel(
    const scalar_t* __restrict__ src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> indptr_info,
    scalar_t* __restrict__ out_data,
    size_t N,
    size_t E) {
  using opmath_t = at::opmath_type<scalar_t>;

  int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_idx >= static_cast<int>(N))
    return;

  // Compute the strided indptr offset for this row.
  int offset = IndexPtrToOffset::get(row_idx, indptr_info);
  int64_t row_start = __ldg(indptr_info.data + offset);
  int64_t row_end = __ldg(indptr_info.data + offset +
                          indptr_info.strides[indptr_info.dims - 1]);

  // Per-batch slice offset into `src` along the reduction axis. Upstream
  // computes this via
  //   `(row_idx / (indptr.size(-1) - 1)) * E`
  // where `E = src.size(dim)`. Mirrors `segment_csr_cuda.cu:38-43`.
  int batch_offset = (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) *
                     static_cast<int>(E);

  opmath_t val = static_cast<opmath_t>(0);
  for (int64_t src_idx = row_start; src_idx < row_end; ++src_idx) {
    val += static_cast<opmath_t>(src_data[batch_offset + src_idx]);
  }

  // Accumulate into `out`. The dispatcher zero-inits a freshly allocated
  // `out`; when the caller supplies `out=`, we **add** to it (accumulate
  // contract used by `GatherCSR::backward`). For empty rows
  // (`row_end == row_start`), `val == 0` so the existing value is preserved.
  out_data[row_idx] =
      static_cast<scalar_t>(static_cast<opmath_t>(out_data[row_idx]) + val);
}

// K>1 broadcast kernel — one thread per (row, col) pair.
template <typename scalar_t>
__global__ void segment_sum_csr_broadcast_cuda_kernel(
    const scalar_t* __restrict__ src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> indptr_info,
    scalar_t* __restrict__ out_data,
    size_t N,
    size_t K,
    size_t E) {
  using opmath_t = at::opmath_type<scalar_t>;

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = thread_idx / static_cast<int>(K);
  int col_idx = thread_idx % static_cast<int>(K);

  if (thread_idx >= static_cast<int>(N * K))
    return;

  int offset = IndexPtrToOffset::get(row_idx, indptr_info);
  int64_t row_start = __ldg(indptr_info.data + offset);
  int64_t row_end = __ldg(indptr_info.data + offset +
                          indptr_info.strides[indptr_info.dims - 1]);

  // Per-batch slice offset into `src` includes the trailing `K` extent.
  // Mirrors `segment_csr_cuda.cu:85`.
  int batch_offset = (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) *
                     static_cast<int>(E) * static_cast<int>(K);

  opmath_t val = static_cast<opmath_t>(0);
  for (int64_t src_idx = row_start; src_idx < row_end; ++src_idx) {
    val += static_cast<opmath_t>(
        src_data[batch_offset + static_cast<int>(K) * src_idx + col_idx]);
  }

  // Accumulate-into contract (see K==1 kernel above).
  out_data[thread_idx] =
      static_cast<scalar_t>(static_cast<opmath_t>(out_data[thread_idx]) + val);
}

at::Tensor segment_sum_csr_kernel(
    const at::Tensor& src,
    const at::Tensor& indptr,
    const std::optional<at::Tensor>& optional_out) {
  TORCH_CHECK(src.is_cuda(),
              "segment_sum_csr (CUDA): src must be a CUDA tensor");
  TORCH_CHECK(indptr.is_cuda(),
              "segment_sum_csr (CUDA): indptr must be a CUDA tensor");
  TORCH_CHECK(src.device() == indptr.device(),
              "segment_sum_csr (CUDA): src and indptr must be on the same "
              "device");
  TORCH_CHECK(src.dim() >= indptr.dim(),
              "segment_sum_csr (CUDA): src.dim() must be >= indptr.dim() (got ",
              src.dim(), " vs ", indptr.dim(), ")");
  TORCH_CHECK(indptr.dim() >= 1,
              "segment_sum_csr (CUDA): indptr must have at least 1 dimension");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

  // CSR convention: `indptr` is `expand`-broadcast up to
  // `src.shape[:indptr.dim()-1]` so the leading batch dims match. The last
  // dim of `indptr` is the row pointer itself and stays at its native size.
  // Mirrors `segment_csr_cuda.cu:108-112`.
  auto sizes = indptr.sizes().vec();
  for (int64_t i = 0; i < indptr.dim() - 1; ++i) {
    sizes[i] = src.size(i);
  }
  auto indptr_b = indptr.expand(sizes);

  // CSR ops fix the reduction axis at the last `indptr` dim.
  const auto dim = indptr_b.dim() - 1;

  auto src_c = src.contiguous();
  auto indptr_c = indptr_b.contiguous();

  at::Tensor out;
  const bool out_was_provided = optional_out.has_value();
  if (out_was_provided) {
    // Accumulate-into contract: caller owns the buffer, no zero-init.
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "segment_sum_csr (CUDA): out.size(", i,
                    ") must match src.size(", i, ")");
      }
    }
    TORCH_CHECK(src_c.numel() == 0 || out.size(dim) == indptr_c.size(dim) - 1,
                "segment_sum_csr (CUDA): out.size(", dim,
                ") must equal indptr.size(-1) - 1 (got ", out.size(dim), " vs ",
                indptr_c.size(dim) - 1, ")");
  } else {
    auto out_sizes = src_c.sizes().vec();
    out_sizes[dim] = std::max<int64_t>(indptr_c.size(dim) - 1, 0);
    // Zero-init so the in-kernel `+=` accumulation produces the correct
    // result for a freshly allocated buffer.
    out = at::zeros(out_sizes, src_c.options());
  }

  if (src_c.numel() == 0) {
    return out;
  }

  // `N` is the total number of rows across all leading batch dims (upstream
  // `segment_csr_cuda.cu:144`). `K` is the product of trailing feature dims
  // past the reduction axis (upstream line 145). `E` is `src.size(dim)`.
  const auto N = out.size(dim) * (indptr_c.numel() / indptr_c.size(-1));
  const auto K = out.numel() / N;
  const auto E = src_c.size(dim);

  auto indptr_info = at::cuda::detail::getTensorInfo<int64_t, int>(indptr_c);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "segment_sum_csr_cuda", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();

        if (K == 1) {
          // One thread per row, sequential inner sum. No SHFL — TB == 1
          // means there's only one lane per row.
          const int T = threads();
          const int B = blocks(static_cast<int>(N));
          segment_sum_csr_cuda_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, indptr_info, out_data, N, E);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          // One thread per (row, col). Sequential row scan in inner loop.
          const int T = threads();
          const int B = blocks(static_cast<int>(N * K));
          segment_sum_csr_broadcast_cuda_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, indptr_info, out_data, N, K, E);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });

  return out;
}

// ============================================================================
// `segment_mean_csr` — Commit 11.
//
// Strategy: reuse the two `segment_sum_csr_*` kernels above for the per-row
// sum pass, then divide by per-row lengths.
//
// Row-length computation — `indptr.diff()` along the last `indptr` dim. This
// is the same data the kernels already load (`row_end - row_start`), but doing
// it as a single tensor op on the host keeps the K==1 / K>1 kernels identical
// to the sum case (no need to thread a `count_data` pointer through). Empty
// rows have `diff == 0`; we `masked_fill_(diff == 0, 1)` so the subsequent
// division leaves the zero-init `out[r]` at `0` — matches the upstream
// reducer's "empty row -> 0" semantic at `segment_csr_cuda.cu:38-43` via
// `Reducer::initial_value()` for MEAN.
//
// Compared to `segment_mean_coo` (which needs an atomic counting kernel
// because COO `index` can repeat arbitrarily), CSR row lengths are trivially
// available from `indptr` — no extra kernel launch.
//
// `out=` contract: matches upstream — divides into a fresh `at::zeros` buffer,
// or into the caller-supplied buffer (where the sum kernel's `+=` accumulates
// before the divide, then we divide the **combined** value by the row length).
// This is the same combined semantic as `segment_mean_coo`.
//
// Dispatch: `AT_DISPATCH_FLOATING_TYPES_AND2` for the floating mean path
// (true division); integer types take the `floor` path. Same convention as
// `segment_mean_coo`.

at::Tensor segment_mean_csr_kernel(
    const at::Tensor& src,
    const at::Tensor& indptr,
    const std::optional<at::Tensor>& optional_out) {
  TORCH_CHECK(src.is_cuda(),
              "segment_mean_csr (CUDA): src must be a CUDA tensor");
  TORCH_CHECK(indptr.is_cuda(),
              "segment_mean_csr (CUDA): indptr must be a CUDA tensor");
  TORCH_CHECK(src.device() == indptr.device(),
              "segment_mean_csr (CUDA): src and indptr must be on the same "
              "device");
  TORCH_CHECK(src.dim() >= indptr.dim(),
              "segment_mean_csr (CUDA): src.dim() must be >= indptr.dim() "
              "(got ",
              src.dim(), " vs ", indptr.dim(), ")");
  TORCH_CHECK(indptr.dim() >= 1,
              "segment_mean_csr (CUDA): indptr must have at least 1 dimension");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

  // Same broadcast rule as `segment_sum_csr` — `indptr` leading dims expand
  // up to `src.shape[:indptr.dim()-1]`, last dim of `indptr` stays as-is.
  auto sizes = indptr.sizes().vec();
  for (int64_t i = 0; i < indptr.dim() - 1; ++i) {
    sizes[i] = src.size(i);
  }
  auto indptr_b = indptr.expand(sizes);

  const auto dim = indptr_b.dim() - 1;

  auto src_c = src.contiguous();
  auto indptr_c = indptr_b.contiguous();

  at::Tensor out;
  const bool out_was_provided = optional_out.has_value();
  if (out_was_provided) {
    // Caller owns the buffer; we accumulate into it during the sum pass and
    // then divide the combined value (mirrors `segment_mean_coo` semantics).
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "segment_mean_csr (CUDA): out.size(", i,
                    ") must match src.size(", i, ")");
      }
    }
    TORCH_CHECK(src_c.numel() == 0 || out.size(dim) == indptr_c.size(dim) - 1,
                "segment_mean_csr (CUDA): out.size(", dim,
                ") must equal indptr.size(-1) - 1 (got ", out.size(dim), " vs ",
                indptr_c.size(dim) - 1, ")");
  } else {
    auto out_sizes = src_c.sizes().vec();
    out_sizes[dim] = std::max<int64_t>(indptr_c.size(dim) - 1, 0);
    // Zero-init so the in-kernel `+=` sum accumulation starts from 0, and
    // empty rows stay at 0 (no division occurs in the kernel; the
    // post-pass divide reads `0 / 1 == 0`).
    out = at::zeros(out_sizes, src_c.options());
  }

  if (src_c.numel() == 0) {
    return out;
  }

  const auto N = out.size(dim) * (indptr_c.numel() / indptr_c.size(-1));
  const auto K = out.numel() / N;
  const auto E = src_c.size(dim);

  auto indptr_info = at::cuda::detail::getTensorInfo<int64_t, int>(indptr_c);
  const auto stream = at::cuda::getCurrentCUDAStream();

  // -------- Pass 1: sum into `out`. Reuses the `segment_sum_csr_*` kernels.
  // Mean uses floating-point div, so dispatch on floating types (+ Half /
  // BFloat16). Mirrors the `segment_mean_coo` dispatch.
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "segment_mean_csr_cuda", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();

        if (K == 1) {
          const int T = threads();
          const int B = blocks(static_cast<int>(N));
          segment_sum_csr_cuda_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, indptr_info, out_data, N, E);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          const int T = threads();
          const int B = blocks(static_cast<int>(N * K));
          segment_sum_csr_broadcast_cuda_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, indptr_info, out_data, N, K, E);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });

  // -------- Pass 2: divide by row lengths.
  //
  // `indptr.diff()` along the last dim gives the per-row source-element
  // count, shape `[..., rows]` (matches `out` along dims `[0, dim]`). Empty
  // rows have count 0; mask them up to 1 so the broadcast divide preserves
  // the zero-init in those slots. Cast to `out.dtype()` so the broadcast
  // matches the output's scalar type. Then unsqueeze trailing K dims to
  // broadcast along the feature axis. Same divide pattern as
  // `segment_mean_coo`.
  auto count = indptr_c.diff(/*n=*/1, /*dim=*/-1).to(out.options());
  count.masked_fill_(count == 0, 1);
  for (int64_t i = dim + 1; i < out.dim(); ++i) {
    count = count.unsqueeze(-1);
  }
  out.true_divide_(count);

  return out;
}

// ============================================================================
// `gather_csr` — Commit 10.
//
// Symmetric inverse of `segment_sum_csr`: for each row `r`, broadcast
// `src[r]` to every output position `[indptr[r], indptr[r+1])` along the
// reduction axis. Two kernel variants:
//
//   * K==1: one thread per row writes the row value to all `[row_start,
//     row_end)` output positions. Port of upstream `segment_csr_cuda.cu:
//     171-192` with `TB == 1`. (Upstream uses TB=4 to get more parallelism
//     for long rows; we mirror that ladder via the dynamic block-size
//     dispatch — see comment below.)
//   * K>1: one thread per `(row, col)` pair, same write pattern with the
//     trailing `K` extent. Port of upstream `segment_csr_cuda.cu:194-217`.
//
// `out=` contract: **overwrite** (not accumulate) — every output element is
// written exactly once. Matches upstream `gather_csr`. The output sizing
// rule allocates the same shape as `src` with the reduction axis replaced
// by `indptr[..., -1]` (upstream lines 245-253).

template <typename scalar_t>
__global__ void gather_csr_cuda_kernel(
    const scalar_t* __restrict__ src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> indptr_info,
    scalar_t* __restrict__ out_data,
    size_t N,
    size_t E) {
  int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_idx >= static_cast<int>(N))
    return;

  int offset = IndexPtrToOffset::get(row_idx, indptr_info);
  int row_start = static_cast<int>(__ldg(indptr_info.data + offset));
  int row_end = static_cast<int>(__ldg(
      indptr_info.data + offset + indptr_info.strides[indptr_info.dims - 1]));
  scalar_t val = __ldg(src_data + row_idx);

  // Per-batch slice offset into `out` along the reduction axis. Mirrors
  // upstream `segment_csr_cuda.cu:187`.
  int batch_offset = (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) *
                     static_cast<int>(E);
  for (int out_idx = row_start; out_idx < row_end; ++out_idx) {
    out_data[batch_offset + out_idx] = val;
  }
}

template <typename scalar_t>
__global__ void gather_csr_broadcast_cuda_kernel(
    const scalar_t* __restrict__ src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> indptr_info,
    scalar_t* __restrict__ out_data,
    size_t N,
    size_t K,
    size_t E) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = thread_idx / static_cast<int>(K);
  int col_idx = thread_idx % static_cast<int>(K);

  if (thread_idx >= static_cast<int>(N * K))
    return;

  int offset = IndexPtrToOffset::get(row_idx, indptr_info);
  int row_start = static_cast<int>(__ldg(indptr_info.data + offset));
  int row_end = static_cast<int>(__ldg(
      indptr_info.data + offset + indptr_info.strides[indptr_info.dims - 1]));
  scalar_t val = src_data[thread_idx];

  int batch_offset = (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) *
                     static_cast<int>(E) * static_cast<int>(K);
  for (int out_idx = row_start; out_idx < row_end; ++out_idx) {
    out_data[batch_offset + static_cast<int>(K) * out_idx + col_idx] = val;
  }
}

at::Tensor gather_csr_kernel(const at::Tensor& src,
                             const at::Tensor& indptr,
                             const std::optional<at::Tensor>& optional_out) {
  TORCH_CHECK(src.is_cuda(), "gather_csr (CUDA): src must be a CUDA tensor");
  TORCH_CHECK(indptr.is_cuda(),
              "gather_csr (CUDA): indptr must be a CUDA tensor");
  TORCH_CHECK(src.device() == indptr.device(),
              "gather_csr (CUDA): src and indptr must be on the same device");
  TORCH_CHECK(src.dim() >= indptr.dim(),
              "gather_csr (CUDA): src.dim() must be >= indptr.dim() (got ",
              src.dim(), " vs ", indptr.dim(), ")");
  TORCH_CHECK(indptr.dim() >= 1,
              "gather_csr (CUDA): indptr must have at least 1 dimension");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

  // Broadcast `indptr` leading dims up to `src.shape[:indptr.dim()-1]`.
  // Mirrors upstream `segment_csr_cuda.cu:229-232`.
  auto sizes = indptr.sizes().vec();
  for (int64_t i = 0; i < indptr.dim() - 1; ++i) {
    sizes[i] = src.size(i);
  }
  auto indptr_b = indptr.expand(sizes);

  const auto dim = indptr_b.dim() - 1;
  TORCH_CHECK(src.size(dim) == 0 || src.size(dim) == indptr_b.size(dim) - 1,
              "gather_csr (CUDA): src.size(", dim,
              ") must be 0 or equal to indptr.size(-1) - 1 (got ",
              src.size(dim), " vs ", indptr_b.size(dim) - 1, ")");

  auto src_c = src.contiguous();
  auto indptr_c = indptr_b.contiguous();

  at::Tensor out;
  if (optional_out.has_value()) {
    // Overwrite contract: every output element is written exactly once.
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "gather_csr (CUDA): out.size(", i, ") must match src.size(",
                    i, ")");
      }
    }
  } else {
    auto out_sizes = src_c.sizes().vec();
    if (src_c.numel() > 0) {
      // `out.size(dim) = indptr[..., -1]` — the total number of "gathered"
      // entries. Mirrors upstream `segment_csr_cuda.cu:247-252`. Reading
      // the last entry of a contiguous flat `indptr` costs one D->H sync
      // but is one-off (host-side allocation path).
      out_sizes[dim] = indptr_c.flatten()[-1].cpu().data_ptr<int64_t>()[0];
    } else {
      out_sizes[dim] = 0;
    }
    out = at::empty(out_sizes, src_c.options());
  }

  if (src_c.numel() == 0) {
    if (!optional_out.has_value()) {
      out.fill_(0);
    }
    return out;
  }

  // `N` = total number of input rows across all batch dims (upstream
  // `segment_csr_cuda.cu:261`). `K` = trailing feature-dim extent. `E` =
  // `out.size(dim)` — the per-batch output length.
  const auto N = src_c.size(dim) * (indptr_c.numel() / indptr_c.size(-1));
  const auto K = src_c.numel() / N;
  const auto E = out.size(dim);

  auto indptr_info = at::cuda::detail::getTensorInfo<int64_t, int>(indptr_c);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "gather_csr_cuda", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();

        if (K == 1) {
          // One thread per row writes the row value to all output positions
          // in `[row_start, row_end)`.
          const int T = threads();
          const int B = blocks(static_cast<int>(N));
          gather_csr_cuda_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, indptr_info, out_data, N, E);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          // One thread per (row, col); same write pattern with the trailing
          // `K` extent.
          const int T = threads();
          const int B = blocks(static_cast<int>(N * K));
          gather_csr_broadcast_cuda_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, indptr_info, out_data, N, K, E);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });

  return out;
}

// ============================================================================
// `segment_min_csr` — Commit 12.
//
// First CSR op to use `TB > 1`: each warp lane (one of 32) collaborates on a
// single row, then a warp-tree `SHFL_DOWN_SYNC` reduction halves the active
// lane count five times to land the row's `(min_value, argindex)` in lane 0.
//
// Why `TB == 1` is wrong for min/max (and was fine for sum/mean):
// sum/mean's inner loop is `+=` and a single thread per row was already fast
// enough — adding TB warp lanes would just waste them on the tail of short
// rows. Min/max, however, has no early-exit and benefits more from
// parallelism over long rows. Upstream `segment_csr_cuda.cu:157` launches
// `<<<BLOCKS(32, N), THREADS>>>` for min/max (32 lanes per row), versus
// `<<<BLOCKS(1, N), THREADS>>>` (TB == 1) for sum.
//
// TB choice: we mirror upstream and fix `TB == 32` (one warp per row). Long
// rows benefit from the parallelism; short rows just leave the trailing
// lanes holding `(numeric_limits::max(), E_dim)` which contributes nothing.
// Picking TB == warp size also lets us drive the reduction with the existing
// `SHFL_DOWN_SYNC` macro without involving shared memory.
//
// Pair propagation through `SHFL_DOWN_SYNC`:
//
//   Each lane holds a local `(val, arg)`. At reduction step `i` (i = 16, 8,
//   4, 2, 1):
//
//     tmp_val = SHFL_DOWN_SYNC(FULL_MASK, val, i);
//     tmp_arg = SHFL_DOWN_SYNC(FULL_MASK, arg, i);
//     if (device_lt(tmp_val, val)) { val = tmp_val; arg = tmp_arg; }
//
//   We perform two independent shuffles — one carrying `val`, one carrying
//   `arg` — and then choose between the **pair members together**, so the
//   surviving `arg` always matches the surviving `val`. After 5 steps (32
//   -> 16 -> 8 -> 4 -> 2 -> 1) lane 0 holds the row's min and corresponding
//   argindex. Mirrors upstream `segment_csr_cuda.cu:46-52`. The comparison
//   is strict `<` (via `device_lt`): on ties, the **lower-lane** value wins,
//   which matches the upstream first-match convention for the argindex.
//
// No shared memory: upstream comment at `segment_csr_cuda.cu:68-70`
// explicitly notes that shared memory was tried and rejected — the
// `__syncthreads` overhead exceeded the benefit. We mirror that decision;
// the warp shuffle is sufficient because TB == warp size.
//
// Argindex semantics: stored as `int64_t` (matches `arg_out` dtype), tracks
// the absolute `src_idx` along the reduction axis (NOT relative to the
// row), which makes it directly indexable into the per-batch slice of `src`
// along `dim`. Upstream stores the same absolute index.
//
// Empty-row post-pass: `out.masked_fill_(arg_out == src.size(dim), 0)`.
// Initial `arg_out` is filled with the sentinel `E_dim = src.size(dim)`;
// any row that ran zero iterations keeps that sentinel, and we reset the
// corresponding `out` slot (still at `numeric_limits::max()`) to 0. Matches
// `segment_min_coo`'s empty-bucket policy and the CPU min kernel contract.

// K==1 kernel — one warp per row, lanes stride through the row by `TB == 32`.
template <typename scalar_t>
__global__ void segment_min_csr_cuda_kernel(
    const scalar_t* __restrict__ src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> indptr_info,
    scalar_t* __restrict__ out_data,
    int64_t* __restrict__ arg_out_data,
    size_t N,
    size_t E) {
  // `TB == WARP_SIZE` (32): one warp processes one row. `lane_idx` selects
  // the slot within the row.
  constexpr int TB = WARP_SIZE;

  const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_idx = thread_idx / TB;
  const int lane_idx = thread_idx & (TB - 1);

  if (row_idx >= static_cast<int>(N))
    return;

  // Strided indptr offset for this row.
  int ip_offset = IndexPtrToOffset::get(row_idx, indptr_info);
  int64_t row_start = __ldg(indptr_info.data + ip_offset);
  int64_t row_end = __ldg(indptr_info.data + ip_offset +
                          indptr_info.strides[indptr_info.dims - 1]);

  // Per-batch slice offset into `src` along the reduction axis. Same as the
  // sum kernel above.
  const int batch_offset =
      (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) *
      static_cast<int>(E);

  // Per-lane sentinel pair. The sentinel argindex `E` (== `src.size(dim)`)
  // matches the host-side `at::full` init of `arg_out`; the host post-pass
  // checks `arg_out == E` to identify empty rows. The sentinel value
  // `numeric_limits<scalar_t>::max()` matches the host-side `out.fill_`.
  scalar_t val = std::numeric_limits<scalar_t>::max();
  int64_t arg = static_cast<int64_t>(E);

  // Lane-strided scan: each lane handles `ceil(row_len / TB)` elements at
  // stride `TB`. Mirrors upstream `segment_csr_cuda.cu:39-43`.
  //
  // On ties (`cand == val`), we keep the **earlier** (lower `src_idx`)
  // winner via strict `<` — matches the CPU kernel's first-match argindex
  // convention. Pathological case: if every element in a row equals
  // `numeric_limits::max()`, `arg` stays at sentinel `E`, and the host
  // post-pass will `masked_fill_` `out` to 0 (matches upstream's behaviour
  // for the same edge case).
  for (int64_t src_idx = row_start + lane_idx; src_idx < row_end;
       src_idx += TB) {
    scalar_t cand = src_data[batch_offset + src_idx];
    if (device_lt(cand, val)) {
      val = cand;
      arg = src_idx;
    }
  }

  // Warp-tree min reduction over `(val, arg)` pairs. Two independent
  // shuffles transport the pair members across lanes; we then choose
  // between them as a unit so the surviving `arg` always matches the
  // surviving `val`. Mirrors upstream `segment_csr_cuda.cu:45-52`.
#pragma unroll
  for (int i = TB / 2; i > 0; i /= 2) {
    scalar_t tmp_val = SHFL_DOWN_SYNC(FULL_MASK, val, i);
    int64_t tmp_arg = SHFL_DOWN_SYNC(FULL_MASK, arg, i);
    // Strict `<`: on ties, keep the existing (lower-lane) value/arg, which
    // matches the first-match convention for the argindex.
    if (device_lt(tmp_val, val)) {
      val = tmp_val;
      arg = tmp_arg;
    }
  }

  // Lane 0 writes the row's `(min, argmin)` pair. Other lanes contribute
  // nothing past this point. No `atomicMin` needed — the warp-tree
  // reduction has already aggregated all of row `r`'s contributions, and
  // each warp owns exactly one row.
  //
  // For empty rows (`row_start == row_end`), the inner loop ran zero
  // iterations, `val` is still `numeric_limits::max()`, and `arg` is still
  // the sentinel `E`. The host post-pass resets such slots to 0.
  if (lane_idx == 0) {
    out_data[row_idx] = val;
    arg_out_data[row_idx] = arg;
  }
}

// K>1 broadcast kernel — one thread per (row, col) pair. Sequential row scan
// with running `(val, arg)`; no shuffle (each thread already owns its full
// reduction). Mirrors upstream `segment_csr_cuda.cu:62-95`.
template <typename scalar_t>
__global__ void segment_min_csr_broadcast_cuda_kernel(
    const scalar_t* __restrict__ src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> indptr_info,
    scalar_t* __restrict__ out_data,
    int64_t* __restrict__ arg_out_data,
    size_t N,
    size_t K,
    size_t E) {
  const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_idx = thread_idx / static_cast<int>(K);
  const int col_idx = thread_idx % static_cast<int>(K);

  if (thread_idx >= static_cast<int>(N * K))
    return;

  int ip_offset = IndexPtrToOffset::get(row_idx, indptr_info);
  int64_t row_start = __ldg(indptr_info.data + ip_offset);
  int64_t row_end = __ldg(indptr_info.data + ip_offset +
                          indptr_info.strides[indptr_info.dims - 1]);

  const int batch_offset =
      (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) *
      static_cast<int>(E) * static_cast<int>(K);

  scalar_t val = std::numeric_limits<scalar_t>::max();
  int64_t arg = static_cast<int64_t>(E);

  for (int64_t src_idx = row_start; src_idx < row_end; ++src_idx) {
    scalar_t cand =
        src_data[batch_offset + static_cast<int>(K) * src_idx + col_idx];
    // Same strict-`<` first-match logic as the K==1 kernel above.
    if (device_lt(cand, val)) {
      val = cand;
      arg = src_idx;
    }
  }

  out_data[thread_idx] = val;
  arg_out_data[thread_idx] = arg;
}

std::tuple<at::Tensor, at::Tensor> segment_min_csr_kernel(
    const at::Tensor& src,
    const at::Tensor& indptr,
    const std::optional<at::Tensor>& optional_out) {
  TORCH_CHECK(src.is_cuda(),
              "segment_min_csr (CUDA): src must be a CUDA tensor");
  TORCH_CHECK(indptr.is_cuda(),
              "segment_min_csr (CUDA): indptr must be a CUDA tensor");
  TORCH_CHECK(src.device() == indptr.device(),
              "segment_min_csr (CUDA): src and indptr must be on the same "
              "device");
  TORCH_CHECK(src.dim() >= indptr.dim(),
              "segment_min_csr (CUDA): src.dim() must be >= indptr.dim() (got ",
              src.dim(), " vs ", indptr.dim(), ")");
  TORCH_CHECK(indptr.dim() >= 1,
              "segment_min_csr (CUDA): indptr must have at least 1 dimension");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

  // Same broadcast rule as `segment_sum_csr`: leading dims of `indptr`
  // expand to `src.shape[:indptr.dim()-1]`; last `indptr` dim is the row
  // pointer itself (kept native).
  auto sizes = indptr.sizes().vec();
  for (int64_t i = 0; i < indptr.dim() - 1; ++i) {
    sizes[i] = src.size(i);
  }
  auto indptr_b = indptr.expand(sizes);

  const auto dim = indptr_b.dim() - 1;

  auto src_c = src.contiguous();
  auto indptr_c = indptr_b.contiguous();

  // `E_dim = src.size(dim)` is the sentinel argindex value for `arg_out`
  // (matches `segment_min_coo` and `scatter_min`).
  const int64_t E_dim = src_c.size(dim);

  at::Tensor out;
  const bool out_was_provided = optional_out.has_value();
  if (out_was_provided) {
    // Caller owns the buffer; no max-init. With `out=` supplied, the kernel
    // **overwrites** the per-row slot with the computed row min — there is
    // no atomicMin path here (the warp-tree fully reduces the row before
    // the single write), so any prior value in `out` is discarded for
    // non-empty rows. This matches upstream's "out= as scratch buffer"
    // contract: the caller is expected to pre-initialize if they want a
    // running min across calls.
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "segment_min_csr (CUDA): out.size(", i,
                    ") must match src.size(", i, ")");
      }
    }
    TORCH_CHECK(src_c.numel() == 0 || out.size(dim) == indptr_c.size(dim) - 1,
                "segment_min_csr (CUDA): out.size(", dim,
                ") must equal indptr.size(-1) - 1 (got ", out.size(dim), " vs ",
                indptr_c.size(dim) - 1, ")");
  } else {
    auto out_sizes = src_c.sizes().vec();
    out_sizes[dim] = std::max<int64_t>(indptr_c.size(dim) - 1, 0);
    // Allocate uninit then fill with per-dtype `max()`. Same rationale as
    // `segment_min_coo`: `at::full` with a single host `Scalar` cannot
    // represent each dtype's max correctly, so we route through
    // `AT_DISPATCH_*` + `numeric_limits<scalar_t>::max()`.
    out = at::empty(out_sizes, src_c.options());
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
        "segment_min_csr_cuda_init_out",
        [&] { out.fill_(std::numeric_limits<scalar_t>::max()); });
  }

  // `arg_out` is always freshly allocated and filled with the sentinel
  // `E_dim`. The kernel either overwrites it with a real argindex (non-empty
  // row) or leaves it at the sentinel (empty row); the host post-pass below
  // uses `arg_out == E_dim` to identify empty rows.
  auto arg_out = at::full(out.sizes(), E_dim,
                          indptr_c.options().dtype(at::ScalarType::Long));

  if (src_c.numel() == 0) {
    // No contributors at all; every row is empty. Reset to 0 (matching the
    // CPU kernel) when we own `out`. With `out=` supplied, the caller's
    // values are preserved.
    if (!out_was_provided) {
      out.zero_();
    }
    return std::make_tuple(out, arg_out);
  }

  const auto N = out.size(dim) * (indptr_c.numel() / indptr_c.size(-1));
  const auto K = out.numel() / N;
  const auto E = src_c.size(dim);

  auto indptr_info = at::cuda::detail::getTensorInfo<int64_t, int>(indptr_c);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "segment_min_csr_cuda", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        auto* arg_out_data = arg_out.data_ptr<int64_t>();

        if (K == 1) {
          // 32 lanes per row (TB == warp size). Total threads = `32 * N`;
          // block size from the dynamic `threads()` helper (always a
          // multiple of 32 for sm_60+).
          constexpr int TB = WARP_SIZE;
          const int T = threads();
          const int B =
              std::max<int>(1, static_cast<int>((TB * N + T - 1) / T));
          segment_min_csr_cuda_kernel<scalar_t><<<B, T, 0, stream>>>(
              src_data, indptr_info, out_data, arg_out_data, N, E);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          // One thread per (row, col) pair; sequential row scan inside the
          // thread, no shuffle.
          const int T = threads();
          const int B = blocks(static_cast<int>(N * K));
          segment_min_csr_broadcast_cuda_kernel<scalar_t><<<B, T, 0, stream>>>(
              src_data, indptr_info, out_data, arg_out_data, N, K, E);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });

  // Empty-row cleanup: where `arg_out == E_dim`, no contributor wrote to
  // that row, so `out` still holds `numeric_limits::max()`. Reset to `0` to
  // match the CPU kernel's contract. Skipped when `out=` is supplied — the
  // caller's pre-existing values in empty rows must be preserved.
  if (!out_was_provided) {
    out.masked_fill_(arg_out == E_dim, 0);
  }

  return std::make_tuple(out, arg_out);
}

// ============================================================================
// `segment_max_csr` — Commit 13.
//
// Symmetric mirror of `segment_min_csr`: same TB == 32 warp-per-row layout,
// same `SHFL_DOWN_SYNC` pair propagation across two independent shuffles
// (one for `val`, one for `arg`), same K>1 broadcast variant with a
// sequential per-thread row scan. The only differences vs the min kernel:
//
//   * Sentinel value is `numeric_limits<scalar_t>::lowest()` (not `max()`),
//     so the first contributing element always wins.
//   * Per-element comparison uses strict `>` (via `device_gt`) — on ties the
//     **earlier** (lower `src_idx`) winner is kept, matching the CPU max
//     kernel's first-match argindex convention.
//   * Warp-tree reduction also uses `device_gt`: when the partner lane's
//     `val` is *greater*, swap both pair members; otherwise keep the
//     current pair.
//
// Empty-row post-pass: identical to min — `out.masked_fill_(arg_out == E_dim,
// 0)` resets the sentinel `lowest()` to 0 for rows with zero contributors.
//
// No new atomics: like min, the warp-tree fully reduces the row before the
// single lane-0 write; no `atomicMax` is involved.

// K==1 kernel — one warp per row, lanes stride through the row by `TB == 32`.
template <typename scalar_t>
__global__ void segment_max_csr_cuda_kernel(
    const scalar_t* __restrict__ src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> indptr_info,
    scalar_t* __restrict__ out_data,
    int64_t* __restrict__ arg_out_data,
    size_t N,
    size_t E) {
  constexpr int TB = WARP_SIZE;

  const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_idx = thread_idx / TB;
  const int lane_idx = thread_idx & (TB - 1);

  if (row_idx >= static_cast<int>(N))
    return;

  int ip_offset = IndexPtrToOffset::get(row_idx, indptr_info);
  int64_t row_start = __ldg(indptr_info.data + ip_offset);
  int64_t row_end = __ldg(indptr_info.data + ip_offset +
                          indptr_info.strides[indptr_info.dims - 1]);

  const int batch_offset =
      (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) *
      static_cast<int>(E);

  // Sentinel pair. `lowest()` for the value (so the first contributor wins
  // against any real value), `E` for the argindex (host-side `at::full` init
  // matches this; host post-pass detects empty rows via `arg_out == E`).
  scalar_t val = std::numeric_limits<scalar_t>::lowest();
  int64_t arg = static_cast<int64_t>(E);

  // Lane-strided scan. Strict `>` keeps the earlier winner on ties,
  // mirroring the CPU max kernel's first-match argindex.
  for (int64_t src_idx = row_start + lane_idx; src_idx < row_end;
       src_idx += TB) {
    scalar_t cand = src_data[batch_offset + src_idx];
    if (device_gt(cand, val)) {
      val = cand;
      arg = src_idx;
    }
  }

  // Warp-tree max reduction over `(val, arg)` pairs. Two independent
  // shuffles transport the pair members across lanes; we swap as a unit so
  // the surviving `arg` always matches the surviving `val`. When the
  // partner's `val` is *greater* than ours we adopt the partner's pair.
#pragma unroll
  for (int i = TB / 2; i > 0; i /= 2) {
    scalar_t tmp_val = SHFL_DOWN_SYNC(FULL_MASK, val, i);
    int64_t tmp_arg = SHFL_DOWN_SYNC(FULL_MASK, arg, i);
    if (device_gt(tmp_val, val)) {
      val = tmp_val;
      arg = tmp_arg;
    }
  }

  // Lane 0 writes the row's `(max, argmax)` pair. For empty rows
  // (`row_start == row_end`), `val` is still `lowest()` and `arg` is still
  // the sentinel `E`; the host post-pass resets those `out` slots to 0.
  if (lane_idx == 0) {
    out_data[row_idx] = val;
    arg_out_data[row_idx] = arg;
  }
}

// K>1 broadcast kernel — one thread per (row, col) pair. Sequential row scan
// with running `(val, arg)`; no shuffle (each thread already owns its full
// reduction).
template <typename scalar_t>
__global__ void segment_max_csr_broadcast_cuda_kernel(
    const scalar_t* __restrict__ src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> indptr_info,
    scalar_t* __restrict__ out_data,
    int64_t* __restrict__ arg_out_data,
    size_t N,
    size_t K,
    size_t E) {
  const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_idx = thread_idx / static_cast<int>(K);
  const int col_idx = thread_idx % static_cast<int>(K);

  if (thread_idx >= static_cast<int>(N * K))
    return;

  int ip_offset = IndexPtrToOffset::get(row_idx, indptr_info);
  int64_t row_start = __ldg(indptr_info.data + ip_offset);
  int64_t row_end = __ldg(indptr_info.data + ip_offset +
                          indptr_info.strides[indptr_info.dims - 1]);

  const int batch_offset =
      (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) *
      static_cast<int>(E) * static_cast<int>(K);

  scalar_t val = std::numeric_limits<scalar_t>::lowest();
  int64_t arg = static_cast<int64_t>(E);

  for (int64_t src_idx = row_start; src_idx < row_end; ++src_idx) {
    scalar_t cand =
        src_data[batch_offset + static_cast<int>(K) * src_idx + col_idx];
    // Same strict-`>` first-match logic as the K==1 kernel above.
    if (device_gt(cand, val)) {
      val = cand;
      arg = src_idx;
    }
  }

  out_data[thread_idx] = val;
  arg_out_data[thread_idx] = arg;
}

std::tuple<at::Tensor, at::Tensor> segment_max_csr_kernel(
    const at::Tensor& src,
    const at::Tensor& indptr,
    const std::optional<at::Tensor>& optional_out) {
  TORCH_CHECK(src.is_cuda(),
              "segment_max_csr (CUDA): src must be a CUDA tensor");
  TORCH_CHECK(indptr.is_cuda(),
              "segment_max_csr (CUDA): indptr must be a CUDA tensor");
  TORCH_CHECK(src.device() == indptr.device(),
              "segment_max_csr (CUDA): src and indptr must be on the same "
              "device");
  TORCH_CHECK(src.dim() >= indptr.dim(),
              "segment_max_csr (CUDA): src.dim() must be >= indptr.dim() (got ",
              src.dim(), " vs ", indptr.dim(), ")");
  TORCH_CHECK(indptr.dim() >= 1,
              "segment_max_csr (CUDA): indptr must have at least 1 dimension");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

  // Same broadcast rule as `segment_sum_csr` / `segment_min_csr`.
  auto sizes = indptr.sizes().vec();
  for (int64_t i = 0; i < indptr.dim() - 1; ++i) {
    sizes[i] = src.size(i);
  }
  auto indptr_b = indptr.expand(sizes);

  const auto dim = indptr_b.dim() - 1;

  auto src_c = src.contiguous();
  auto indptr_c = indptr_b.contiguous();

  const int64_t E_dim = src_c.size(dim);

  at::Tensor out;
  const bool out_was_provided = optional_out.has_value();
  if (out_was_provided) {
    // Caller owns the buffer; no lowest-init. With `out=` supplied, the
    // kernel **overwrites** the per-row slot with the computed row max —
    // same contract as `segment_min_csr` (the warp-tree fully reduces the
    // row before the single write, so any prior value in `out` is
    // discarded for non-empty rows).
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "segment_max_csr (CUDA): out.size(", i,
                    ") must match src.size(", i, ")");
      }
    }
    TORCH_CHECK(src_c.numel() == 0 || out.size(dim) == indptr_c.size(dim) - 1,
                "segment_max_csr (CUDA): out.size(", dim,
                ") must equal indptr.size(-1) - 1 (got ", out.size(dim), " vs ",
                indptr_c.size(dim) - 1, ")");
  } else {
    auto out_sizes = src_c.sizes().vec();
    out_sizes[dim] = std::max<int64_t>(indptr_c.size(dim) - 1, 0);
    // Allocate uninit then fill with per-dtype `lowest()`. Same rationale
    // as `segment_min_csr`: `at::full` with a single host `Scalar` cannot
    // represent each dtype's lowest correctly, so we route through
    // `AT_DISPATCH_*` + `numeric_limits<scalar_t>::lowest()`.
    out = at::empty(out_sizes, src_c.options());
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
        "segment_max_csr_cuda_init_out",
        [&] { out.fill_(std::numeric_limits<scalar_t>::lowest()); });
  }

  auto arg_out = at::full(out.sizes(), E_dim,
                          indptr_c.options().dtype(at::ScalarType::Long));

  if (src_c.numel() == 0) {
    if (!out_was_provided) {
      out.zero_();
    }
    return std::make_tuple(out, arg_out);
  }

  const auto N = out.size(dim) * (indptr_c.numel() / indptr_c.size(-1));
  const auto K = out.numel() / N;
  const auto E = src_c.size(dim);

  auto indptr_info = at::cuda::detail::getTensorInfo<int64_t, int>(indptr_c);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "segment_max_csr_cuda", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        auto* arg_out_data = arg_out.data_ptr<int64_t>();

        if (K == 1) {
          constexpr int TB = WARP_SIZE;
          const int T = threads();
          const int B =
              std::max<int>(1, static_cast<int>((TB * N + T - 1) / T));
          segment_max_csr_cuda_kernel<scalar_t><<<B, T, 0, stream>>>(
              src_data, indptr_info, out_data, arg_out_data, N, E);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          const int T = threads();
          const int B = blocks(static_cast<int>(N * K));
          segment_max_csr_broadcast_cuda_kernel<scalar_t><<<B, T, 0, stream>>>(
              src_data, indptr_info, out_data, arg_out_data, N, K, E);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });

  // Empty-row cleanup: where `arg_out == E_dim`, no contributor wrote, so
  // `out` still holds `lowest()`. Reset to `0` to match the CPU kernel's
  // contract. Skipped when `out=` was caller-provided.
  if (!out_was_provided) {
    out.masked_fill_(arg_out == E_dim, 0);
  }

  return std::make_tuple(out, arg_out);
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_sum_csr"),
         TORCH_FN(segment_sum_csr_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_mean_csr"),
         TORCH_FN(segment_mean_csr_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_min_csr"),
         TORCH_FN(segment_min_csr_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_max_csr"),
         TORCH_FN(segment_max_csr_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::gather_csr"), TORCH_FN(gather_csr_kernel));
}

}  // namespace ops
}  // namespace pyg
