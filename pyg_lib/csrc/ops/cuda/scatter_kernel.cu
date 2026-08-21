#include "../scatter.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <limits>
#include <tuple>
#include <type_traits>

#include "atomics.cuh"

namespace pyg {
namespace ops {

namespace {

// Convention 13: WARP_SIZE is fixed at 32 on every NVIDIA architecture pyg-lib
// targets (sm_60+). ROCm warp=64 is not a target. Not used by this commit;
// included here so subsequent commits in this family can rely on the same
// constant without redefining it.
#define WARP_SIZE 32

// Convention 12: dynamic block sizing. Replicates the inline `threads()` /
// `blocks()` pattern from
// `pyg_lib/csrc/sampler/cuda/random_walk_kernel.cu:10-21` — there is no shared
// helper in `pyg_lib/csrc/ops/cuda/` yet, so each new kernel file carries its
// own copy. `maxThreadsPerBlock` is capped at 1024 because launching with more
// threads than `maxThreadsPerBlock` would fail at runtime, and 1024 is the
// upper bound on every supported arch.
int threads() {
  const auto props = at::cuda::getCurrentDeviceProperties();
  return std::min(props->maxThreadsPerBlock, 1024);
}

int blocks(int numel) {
  const auto props = at::cuda::getCurrentDeviceProperties();
  const auto blocks_per_sm = props->maxThreadsPerMultiProcessor / 256;
  const auto max_blocks = props->multiProcessorCount * blocks_per_sm;
  const auto max_threads = threads();
  // `std::max(1, ...)` guards against pathological cases where the
  // `cudaDeviceProp` struct is degenerate (e.g. local nvcc/runtime version
  // mismatches that leave `multiProcessorCount` reading as 0/1 due to layout
  // skew). Launching with `<<<0, ...>>>` is a `cudaErrorInvalidValue`; a
  // single block is always a safe fallback for the work we have to do.
  return std::max(
      1, std::min(max_blocks, (numel + max_threads - 1) / max_threads));
}

// 1 thread per `src` element. Atomically accumulates into
// `out[b * N * K + idx * K + k]` where `idx = index_data[thread_idx]`. The
// index tensor must already be broadcast to `src.shape` and `.contiguous()`
// (the dispatcher enforces both). Grid is a strided 1-D loop so we cover
// `numel` correctly even when `numel > gridDim.x * blockDim.x`.
template <typename scalar_t>
__global__ void scatter_sum_cuda_kernel(const scalar_t* __restrict__ src_data,
                                        const int64_t* __restrict__ index_data,
                                        scalar_t* __restrict__ out_data,
                                        int64_t E,
                                        int64_t K,
                                        int64_t N,
                                        int64_t numel) {
  for (int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
       thread_idx < numel; thread_idx += blockDim.x * gridDim.x) {
    const int64_t b = thread_idx / (E * K);
    const int64_t k = thread_idx % K;
    const int64_t idx = index_data[thread_idx];
    atomicAdd(out_data + b * N * K + idx * K + k, src_data[thread_idx]);
  }
}

// CUDA implementation of `pyg::scatter_sum`. Mirrors the CPU kernel layout
// (B, E, K) and the `out=` contract:
//   * `optional_out` provided -> accumulate into the caller's buffer (no zero
//     init). The buffer is `.contiguous()`-ified at the kernel boundary.
//   * `optional_out` absent -> allocate `at::zeros(...)` so that
//     `atomicAdd`-based accumulation starts from a clean slate.
at::Tensor scatter_sum_kernel(const at::Tensor& src,
                              const at::Tensor& index,
                              int64_t dim,
                              const std::optional<at::Tensor>& optional_out,
                              std::optional<int64_t> dim_size) {
  TORCH_CHECK(src.is_cuda(), "scatter_sum (CUDA): src must be a CUDA tensor");
  TORCH_CHECK(index.is_cuda(),
              "scatter_sum (CUDA): index must be a CUDA tensor");
  TORCH_CHECK(src.device() == index.device(),
              "scatter_sum (CUDA): src and index must be on the same device");
  TORCH_CHECK(src.dim() == index.dim(),
              "scatter_sum (CUDA): src.dim() must equal index.dim() after "
              "broadcasting (got src.dim()=",
              src.dim(), ", index.dim()=", index.dim(), ")");

  dim = dim < 0 ? src.dim() + dim : dim;
  TORCH_CHECK(dim >= 0 && dim < src.dim(),
              "scatter_sum (CUDA): dim out of range");

  // Pin the current CUDA device to the input's device so subsequent
  // `at::cuda::getCurrentDeviceProperties()` / `getCurrentCUDAStream()` calls
  // resolve to the right device (matches upstream pytorch_scatter's
  // `c10::cuda::MaybeSetDevice(src.get_device())`).
  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

  // Convention 16: `.contiguous()` at the kernel boundary. The autograd
  // wrapper calls `broadcast(index, src, dim)` (which produces an expanded,
  // non-contiguous view); we materialize it here so the kernel can index by a
  // single linear offset.
  auto src_c = src.contiguous();
  auto index_c = index.contiguous();

  at::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "scatter_sum (CUDA): out.size(", i,
                    ") must match src.size(", i, ")");
      }
    }
  } else {
    auto sizes = src_c.sizes().vec();
    if (dim_size.has_value()) {
      sizes[dim] = dim_size.value();
    } else if (index_c.numel() == 0) {
      sizes[dim] = 0;
    } else {
      sizes[dim] = 1 + index_c.max().cpu().data_ptr<int64_t>()[0];
    }
    // Zero-init so the atomicAdd accumulation starts from a clean slate.
    out = at::zeros(sizes, src_c.options());
  }

  if (src_c.numel() == 0) {
    return out;
  }

  int64_t B = 1;
  for (int64_t i = 0; i < dim; ++i)
    B *= src_c.size(i);
  const int64_t E = src_c.size(dim);
  const int64_t K = src_c.numel() / (B * E);
  const int64_t N = out.size(dim);
  const int64_t numel = src_c.numel();

  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "scatter_sum_cuda", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        const auto* index_data = index_c.data_ptr<int64_t>();
        auto* out_data = out.data_ptr<scalar_t>();

        scatter_sum_cuda_kernel<scalar_t>
            <<<blocks((int)numel), threads(), 0, stream>>>(
                src_data, index_data, out_data, E, K, N, numel);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return out;
}

// 1 thread per `src` element. Atomically multiplies into
// `out[b * N * K + idx * K + k]` where `idx = index_data[thread_idx]`. Same
// addressing scheme as `scatter_sum_cuda_kernel` — only the atomic differs.
template <typename scalar_t>
__global__ void scatter_mul_cuda_kernel(const scalar_t* __restrict__ src_data,
                                        const int64_t* __restrict__ index_data,
                                        scalar_t* __restrict__ out_data,
                                        int64_t E,
                                        int64_t K,
                                        int64_t N,
                                        int64_t numel) {
  for (int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
       thread_idx < numel; thread_idx += blockDim.x * gridDim.x) {
    const int64_t b = thread_idx / (E * K);
    const int64_t k = thread_idx % K;
    const int64_t idx = index_data[thread_idx];
    atomicMul(out_data + b * N * K + idx * K + k, src_data[thread_idx]);
  }
}

// CUDA implementation of `pyg::scatter_mul`. Mirrors `scatter_sum_kernel` with
// two differences:
//   * Uses `atomicMul` (CAS-loop from `atomics.cuh`) for accumulation.
//   * When `optional_out` is absent, the fresh buffer is filled with **ones**,
//     not zeros — multiplicative identity. Mirrors upstream pytorch_scatter's
//     `out.fill_(Reducer<scalar_t, REDUCE>::init())` for `REDUCE = MUL`.
at::Tensor scatter_mul_kernel(const at::Tensor& src,
                              const at::Tensor& index,
                              int64_t dim,
                              const std::optional<at::Tensor>& optional_out,
                              std::optional<int64_t> dim_size) {
  TORCH_CHECK(src.is_cuda(), "scatter_mul (CUDA): src must be a CUDA tensor");
  TORCH_CHECK(index.is_cuda(),
              "scatter_mul (CUDA): index must be a CUDA tensor");
  TORCH_CHECK(src.device() == index.device(),
              "scatter_mul (CUDA): src and index must be on the same device");
  TORCH_CHECK(src.dim() == index.dim(),
              "scatter_mul (CUDA): src.dim() must equal index.dim() after "
              "broadcasting (got src.dim()=",
              src.dim(), ", index.dim()=", index.dim(), ")");

  dim = dim < 0 ? src.dim() + dim : dim;
  TORCH_CHECK(dim >= 0 && dim < src.dim(),
              "scatter_mul (CUDA): dim out of range");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

  // Convention 16: `.contiguous()` at the kernel boundary. The autograd
  // wrapper passes a `broadcast(index, src, dim)`-expanded (non-contiguous)
  // view; materialize it so the kernel can index by a single linear offset.
  auto src_c = src.contiguous();
  auto index_c = index.contiguous();

  at::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "scatter_mul (CUDA): out.size(", i,
                    ") must match src.size(", i, ")");
      }
    }
  } else {
    auto sizes = src_c.sizes().vec();
    if (dim_size.has_value()) {
      sizes[dim] = dim_size.value();
    } else if (index_c.numel() == 0) {
      sizes[dim] = 0;
    } else {
      sizes[dim] = 1 + index_c.max().cpu().data_ptr<int64_t>()[0];
    }
    // Multiplicative identity (1) so `atomicMul` accumulation starts cleanly.
    out = at::ones(sizes, src_c.options());
  }

  if (src_c.numel() == 0) {
    return out;
  }

  int64_t B = 1;
  for (int64_t i = 0; i < dim; ++i)
    B *= src_c.size(i);
  const int64_t E = src_c.size(dim);
  const int64_t K = src_c.numel() / (B * E);
  const int64_t N = out.size(dim);
  const int64_t numel = src_c.numel();

  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "scatter_mul_cuda", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        const auto* index_data = index_c.data_ptr<int64_t>();
        auto* out_data = out.data_ptr<scalar_t>();

        scatter_mul_cuda_kernel<scalar_t>
            <<<blocks((int)numel), threads(), 0, stream>>>(
                src_data, index_data, out_data, E, K, N, numel);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return out;
}

// ============================================================================
// `scatter_min` — Commit 4.
//
// Two CUDA kernels, launched on the same stream so completion ordering is
// implicit:
//
//   1. `scatter_min_value_cuda_kernel`. 1 thread per `src` element; performs
//      `atomicMin(out + linear_offset, src[i])`. `out` is pre-initialized on
//      the host to `numeric_limits<scalar_t>::max()` (when `out=None`); the
//      caller is responsible for any non-default starting state when `out=`
//      is supplied (mirrors the upstream `out.fill_(Reducer::init())` skip).
//
//   2. `scatter_arg_cuda_kernel`. 1 thread per `src` element; re-scans `src`
//      and where `src[i] == out[linear_offset]`, performs
//      `atomicMin(arg_out + linear_offset, i)` over `int64_t`. Initialising
//      `arg_out` to the sentinel `src.size(dim)` makes any valid contributor
//      `i < src.size(dim)` win the CAS; subsequent ties pick the *smallest*
//      `i` (the "first match" argindex convention shared with the CPU
//      kernel — note upstream's CUDA kernel races and returns *some* arg
//      index, but mirroring CPU semantics on the GPU is the conservative
//      port choice and costs only the int64_t atomic).
//
// After both kernels complete, the host clears the sentinel buckets via
// `out.masked_fill_(arg_out == src.size(dim), 0)` so that empty buckets
// produce `0` instead of `numeric_limits::max()`. This is skipped when
// `out=` is supplied — the caller owns the buffer.

// Value-pass kernel.
template <typename scalar_t>
__global__ void scatter_min_value_cuda_kernel(
    const scalar_t* __restrict__ src_data,
    const int64_t* __restrict__ index_data,
    scalar_t* __restrict__ out_data,
    int64_t E,
    int64_t K,
    int64_t N,
    int64_t numel) {
  for (int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
       thread_idx < numel; thread_idx += blockDim.x * gridDim.x) {
    const int64_t b = thread_idx / (E * K);
    const int64_t k = thread_idx % K;
    const int64_t idx = index_data[thread_idx];
    atomicMin(out_data + b * N * K + idx * K + k, src_data[thread_idx]);
  }
}

// Arg-pass kernel. Reads `out_data` (now holding the per-bucket minima from
// the value pass) and atomically lowers `arg_out_data[linear_offset]` to `e`
// (the position within the reduction axis) whenever `src[i] == out[j]`.
template <typename scalar_t>
__global__ void scatter_min_arg_cuda_kernel(
    const scalar_t* __restrict__ src_data,
    const int64_t* __restrict__ index_data,
    const scalar_t* __restrict__ out_data,
    int64_t* __restrict__ arg_out_data,
    int64_t E,
    int64_t K,
    int64_t N,
    int64_t numel) {
  for (int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
       thread_idx < numel; thread_idx += blockDim.x * gridDim.x) {
    const int64_t b = thread_idx / (E * K);
    const int64_t e = (thread_idx / K) % E;
    const int64_t k = thread_idx % K;
    const int64_t idx = index_data[thread_idx];
    const int64_t j = b * N * K + idx * K + k;
    // Equality compared bit-exactly. For `at::Half` / `at::BFloat16`, the
    // `__half`/`__nv_bfloat16` operator overloads from `cuda_fp16.hpp` /
    // `cuda_bf16.hpp` conflict with the built-in `arithmetic ==` once
    // `c10::Half` is convertible to both — so we cast through the underlying
    // bit pattern (`.x`) for half-precision types and use the natural
    // operator for all other dtypes. Bit-exact equality is what we want
    // here: the value pass wrote a specific bit pattern, and we are
    // checking whether `src[i]` has that same bit pattern.
    bool match;
    if constexpr (std::is_same_v<scalar_t, at::Half> ||
                  std::is_same_v<scalar_t, at::BFloat16>) {
      match = src_data[thread_idx].x == out_data[j].x;
    } else {
      match = src_data[thread_idx] == out_data[j];
    }
    if (match) {
      atomicMin(arg_out_data + j, e);
    }
  }
}

std::tuple<at::Tensor, at::Tensor> scatter_min_kernel(
    const at::Tensor& src,
    const at::Tensor& index,
    int64_t dim,
    const std::optional<at::Tensor>& optional_out,
    std::optional<int64_t> dim_size) {
  TORCH_CHECK(src.is_cuda(), "scatter_min (CUDA): src must be a CUDA tensor");
  TORCH_CHECK(index.is_cuda(),
              "scatter_min (CUDA): index must be a CUDA tensor");
  TORCH_CHECK(src.device() == index.device(),
              "scatter_min (CUDA): src and index must be on the same device");
  TORCH_CHECK(src.dim() == index.dim(),
              "scatter_min (CUDA): src.dim() must equal index.dim() after "
              "broadcasting (got src.dim()=",
              src.dim(), ", index.dim()=", index.dim(), ")");

  dim = dim < 0 ? src.dim() + dim : dim;
  TORCH_CHECK(dim >= 0 && dim < src.dim(),
              "scatter_min (CUDA): dim out of range");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

  // Convention 16: `.contiguous()` at the kernel boundary.
  auto src_c = src.contiguous();
  auto index_c = index.contiguous();

  const int64_t E = src_c.size(dim);

  at::Tensor out;
  const bool out_was_provided = optional_out.has_value();
  if (out_was_provided) {
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "scatter_min (CUDA): out.size(", i,
                    ") must match src.size(", i, ")");
      }
    }
  } else {
    auto sizes = src_c.sizes().vec();
    if (dim_size.has_value()) {
      sizes[dim] = dim_size.value();
    } else if (index_c.numel() == 0) {
      sizes[dim] = 0;
    } else {
      sizes[dim] = 1 + index_c.max().cpu().data_ptr<int64_t>()[0];
    }
    // Allocate uninitialized, then fill per-dtype with
    // `std::numeric_limits<scalar_t>::max()`. We can't use `at::full` with
    // a single host `Scalar` because the conversion of `f64 max` to e.g.
    // int8_t would saturate/overflow; the per-dtype dispatch picks the
    // correct `max()` for each scalar. (`std::numeric_limits` has
    // specializations for `at::Half` and `at::BFloat16` provided by
    // PyTorch via `<c10/util/Half.h>` / `<c10/util/BFloat16.h>`.)
    out = at::empty(sizes, src_c.options());
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
        "scatter_min_cuda_init_out",
        [&] { out.fill_(std::numeric_limits<scalar_t>::max()); });
  }

  // `arg_out` is always allocated and always initialized to the sentinel
  // `E = src.size(dim)`. Even when `out=` is supplied, `arg_out` is fresh
  // (the schema returns it; the caller has no way to pre-allocate it).
  auto arg_out =
      at::full(out.sizes(), E, index_c.options().dtype(at::ScalarType::Long));

  if (src_c.numel() == 0) {
    if (!out_was_provided) {
      // No contributors at all: every bucket is "empty" -> sentinel-mask to 0.
      out.zero_();
    }
    return std::make_tuple(out, arg_out);
  }

  int64_t B = 1;
  for (int64_t i = 0; i < dim; ++i)
    B *= src_c.size(i);
  const int64_t K = src_c.numel() / (B * E);
  const int64_t N = out.size(dim);
  const int64_t numel = src_c.numel();

  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "scatter_min_cuda", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        const auto* index_data = index_c.data_ptr<int64_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        auto* arg_out_data = arg_out.data_ptr<int64_t>();

        // Value pass.
        scatter_min_value_cuda_kernel<scalar_t>
            <<<blocks((int)numel), threads(), 0, stream>>>(
                src_data, index_data, out_data, E, K, N, numel);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // Arg pass. Same stream -> the value-pass writes are visible.
        scatter_min_arg_cuda_kernel<scalar_t>
            <<<blocks((int)numel), threads(), 0, stream>>>(
                src_data, index_data, out_data, arg_out_data, E, K, N, numel);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  // Empty-bucket cleanup: where `arg_out == E`, no contributor wrote to that
  // bucket, so `out` still holds `numeric_limits::max()`. Reset to `0` to
  // match the CPU kernel's contract. Only do this when we allocated `out`
  // ourselves — if `out=` was provided, the caller's pre-existing values in
  // empty buckets must be preserved.
  if (!out_was_provided) {
    out.masked_fill_(arg_out == E, 0);
  }

  return std::make_tuple(out, arg_out);
}

// ============================================================================
// `scatter_max` — Commit 5.
//
// Symmetric to `scatter_min` above. Differences:
//   * `out` is initialized to `numeric_limits<scalar_t>::lowest()` (instead of
//     `::max()`).
//   * Value pass uses `atomicMax` (instead of `atomicMin`).
//   * Arg pass logic is *identical* to `scatter_min`: `atomicMin` on the
//     argindex picks the smallest argindex on ties (matches upstream
//     pytorch_scatter's deterministic first-match semantics on CUDA, and is
//     consistent with the CPU kernel).

// Value-pass kernel.
template <typename scalar_t>
__global__ void scatter_max_value_cuda_kernel(
    const scalar_t* __restrict__ src_data,
    const int64_t* __restrict__ index_data,
    scalar_t* __restrict__ out_data,
    int64_t E,
    int64_t K,
    int64_t N,
    int64_t numel) {
  for (int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
       thread_idx < numel; thread_idx += blockDim.x * gridDim.x) {
    const int64_t b = thread_idx / (E * K);
    const int64_t k = thread_idx % K;
    const int64_t idx = index_data[thread_idx];
    atomicMax(out_data + b * N * K + idx * K + k, src_data[thread_idx]);
  }
}

// Arg-pass kernel. Identical to `scatter_min_arg_cuda_kernel` — reads
// `out_data` (now holding per-bucket maxima) and atomically lowers
// `arg_out_data[linear_offset]` to `e` whenever `src[i] == out[j]`. The
// `atomicMin` on the argindex picks the smallest contributor `e` on ties.
template <typename scalar_t>
__global__ void scatter_max_arg_cuda_kernel(
    const scalar_t* __restrict__ src_data,
    const int64_t* __restrict__ index_data,
    const scalar_t* __restrict__ out_data,
    int64_t* __restrict__ arg_out_data,
    int64_t E,
    int64_t K,
    int64_t N,
    int64_t numel) {
  for (int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
       thread_idx < numel; thread_idx += blockDim.x * gridDim.x) {
    const int64_t b = thread_idx / (E * K);
    const int64_t e = (thread_idx / K) % E;
    const int64_t k = thread_idx % K;
    const int64_t idx = index_data[thread_idx];
    const int64_t j = b * N * K + idx * K + k;
    // See `scatter_min_arg_cuda_kernel` for the half-precision bit-pattern
    // equality rationale.
    bool match;
    if constexpr (std::is_same_v<scalar_t, at::Half> ||
                  std::is_same_v<scalar_t, at::BFloat16>) {
      match = src_data[thread_idx].x == out_data[j].x;
    } else {
      match = src_data[thread_idx] == out_data[j];
    }
    if (match) {
      atomicMin(arg_out_data + j, e);
    }
  }
}

std::tuple<at::Tensor, at::Tensor> scatter_max_kernel(
    const at::Tensor& src,
    const at::Tensor& index,
    int64_t dim,
    const std::optional<at::Tensor>& optional_out,
    std::optional<int64_t> dim_size) {
  TORCH_CHECK(src.is_cuda(), "scatter_max (CUDA): src must be a CUDA tensor");
  TORCH_CHECK(index.is_cuda(),
              "scatter_max (CUDA): index must be a CUDA tensor");
  TORCH_CHECK(src.device() == index.device(),
              "scatter_max (CUDA): src and index must be on the same device");
  TORCH_CHECK(src.dim() == index.dim(),
              "scatter_max (CUDA): src.dim() must equal index.dim() after "
              "broadcasting (got src.dim()=",
              src.dim(), ", index.dim()=", index.dim(), ")");

  dim = dim < 0 ? src.dim() + dim : dim;
  TORCH_CHECK(dim >= 0 && dim < src.dim(),
              "scatter_max (CUDA): dim out of range");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

  // Convention 16: `.contiguous()` at the kernel boundary.
  auto src_c = src.contiguous();
  auto index_c = index.contiguous();

  const int64_t E = src_c.size(dim);

  at::Tensor out;
  const bool out_was_provided = optional_out.has_value();
  if (out_was_provided) {
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "scatter_max (CUDA): out.size(", i,
                    ") must match src.size(", i, ")");
      }
    }
  } else {
    auto sizes = src_c.sizes().vec();
    if (dim_size.has_value()) {
      sizes[dim] = dim_size.value();
    } else if (index_c.numel() == 0) {
      sizes[dim] = 0;
    } else {
      sizes[dim] = 1 + index_c.max().cpu().data_ptr<int64_t>()[0];
    }
    // Allocate uninitialized, then fill per-dtype with
    // `std::numeric_limits<scalar_t>::lowest()`. Symmetric to `scatter_min`'s
    // `::max()` init — see that block for the rationale on per-dtype dispatch.
    out = at::empty(sizes, src_c.options());
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
        "scatter_max_cuda_init_out",
        [&] { out.fill_(std::numeric_limits<scalar_t>::lowest()); });
  }

  // `arg_out` is always allocated and always initialized to the sentinel
  // `E = src.size(dim)`. Same convention as `scatter_min`.
  auto arg_out =
      at::full(out.sizes(), E, index_c.options().dtype(at::ScalarType::Long));

  if (src_c.numel() == 0) {
    if (!out_was_provided) {
      // No contributors at all: every bucket is "empty" -> sentinel-mask to 0.
      out.zero_();
    }
    return std::make_tuple(out, arg_out);
  }

  int64_t B = 1;
  for (int64_t i = 0; i < dim; ++i)
    B *= src_c.size(i);
  const int64_t K = src_c.numel() / (B * E);
  const int64_t N = out.size(dim);
  const int64_t numel = src_c.numel();

  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "scatter_max_cuda", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        const auto* index_data = index_c.data_ptr<int64_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        auto* arg_out_data = arg_out.data_ptr<int64_t>();

        // Value pass.
        scatter_max_value_cuda_kernel<scalar_t>
            <<<blocks((int)numel), threads(), 0, stream>>>(
                src_data, index_data, out_data, E, K, N, numel);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // Arg pass. Same stream -> the value-pass writes are visible.
        scatter_max_arg_cuda_kernel<scalar_t>
            <<<blocks((int)numel), threads(), 0, stream>>>(
                src_data, index_data, out_data, arg_out_data, E, K, N, numel);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  // Empty-bucket cleanup: where `arg_out == E`, no contributor wrote to that
  // bucket, so `out` still holds `numeric_limits::lowest()`. Reset to `0` to
  // match the CPU kernel's contract. Skipped when `out=` was supplied.
  if (!out_was_provided) {
    out.masked_fill_(arg_out == E, 0);
  }

  return std::make_tuple(out, arg_out);
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::scatter_sum"),
         TORCH_FN(scatter_sum_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::scatter_mul"),
         TORCH_FN(scatter_mul_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::scatter_min"),
         TORCH_FN(scatter_min_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::scatter_max"),
         TORCH_FN(scatter_max_kernel));
}

}  // namespace ops
}  // namespace pyg
