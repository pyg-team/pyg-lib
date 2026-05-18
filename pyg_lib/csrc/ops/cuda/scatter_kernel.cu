#include "../scatter.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

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

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::scatter_sum"),
         TORCH_FN(scatter_sum_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::scatter_mul"),
         TORCH_FN(scatter_mul_kernel));
}

}  // namespace ops
}  // namespace pyg
