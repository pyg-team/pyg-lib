#include "../segment_csr.h"

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

namespace pyg {
namespace ops {

namespace {

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

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_sum_csr"),
         TORCH_FN(segment_sum_csr_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_mean_csr"),
         TORCH_FN(segment_mean_csr_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::gather_csr"), TORCH_FN(gather_csr_kernel));
}

}  // namespace ops
}  // namespace pyg
