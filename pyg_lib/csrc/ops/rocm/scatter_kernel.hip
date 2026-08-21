#include "../scatter.h"

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

#include <limits>
#include <tuple>
#include <type_traits>

#include "atomics.cuh"

namespace pyg {
namespace ops {

namespace {

// WARP_SIZE is fixed at 32 on every architecture pyg-lib targets.
#define WARP_SIZE 32

// Dynamic block sizing.
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

template <typename scalar_t>
__global__ void scatter_sum_hip_kernel(const scalar_t* __restrict__ src_data,
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

at::Tensor scatter_sum_kernel(const at::Tensor& src,
                              const at::Tensor& index,
                              int64_t dim,
                              const std::optional<at::Tensor>& optional_out,
                              std::optional<int64_t> dim_size) {
  TORCH_CHECK(src.is_cuda(), "scatter_sum (HIP): src must be a CUDA tensor");
  TORCH_CHECK(index.is_cuda(),
              "scatter_sum (HIP): index must be a CUDA tensor");
  TORCH_CHECK(src.device() == index.device(),
              "scatter_sum (HIP): src and index must be on the same device");
  TORCH_CHECK(src.dim() == index.dim(),
              "scatter_sum (HIP): src.dim() must equal index.dim() after "
              "broadcasting (got src.dim()=",
              src.dim(), ", index.dim()=", index.dim(), ")");

  dim = dim < 0 ? src.dim() + dim : dim;
  TORCH_CHECK(dim >= 0 && dim < src.dim(),
              "scatter_sum (HIP): dim out of range");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

  auto src_c = src.contiguous();
  auto index_c = index.contiguous();

  at::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "scatter_sum (HIP): out.size(", i,
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
      "scatter_sum_hip", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        const auto* index_data = index_c.data_ptr<int64_t>();
        auto* out_data = out.data_ptr<scalar_t>();

        scatter_sum_hip_kernel<scalar_t>
            <<<blocks((int)numel), threads(), 0, stream>>>(
                src_data, index_data, out_data, E, K, N, numel);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return out;
}

template <typename scalar_t>
__global__ void scatter_mul_hip_kernel(const scalar_t* __restrict__ src_data,
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

at::Tensor scatter_mul_kernel(const at::Tensor& src,
                              const at::Tensor& index,
                              int64_t dim,
                              const std::optional<at::Tensor>& optional_out,
                              std::optional<int64_t> dim_size) {
  TORCH_CHECK(src.is_cuda(), "scatter_mul (HIP): src must be a CUDA tensor");
  TORCH_CHECK(index.is_cuda(),
              "scatter_mul (HIP): index must be a CUDA tensor");
  TORCH_CHECK(src.device() == index.device(),
              "scatter_mul (HIP): src and index must be on the same device");
  TORCH_CHECK(src.dim() == index.dim(),
              "scatter_mul (HIP): src.dim() must equal index.dim() after "
              "broadcasting (got src.dim()=",
              src.dim(), ", index.dim()=", index.dim(), ")");

  dim = dim < 0 ? src.dim() + dim : dim;
  TORCH_CHECK(dim >= 0 && dim < src.dim(),
              "scatter_mul (HIP): dim out of range");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

  auto src_c = src.contiguous();
  auto index_c = index.contiguous();

  at::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "scatter_mul (HIP): out.size(", i,
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
      "scatter_mul_hip", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        const auto* index_data = index_c.data_ptr<int64_t>();
        auto* out_data = out.data_ptr<scalar_t>();

        scatter_mul_hip_kernel<scalar_t>
            <<<blocks((int)numel), threads(), 0, stream>>>(
                src_data, index_data, out_data, E, K, N, numel);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return out;
}

// Value-pass kernel.
template <typename scalar_t>
__global__ void scatter_min_value_hip_kernel(
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

// Arg-pass kernel.
template <typename scalar_t>
__global__ void scatter_min_arg_hip_kernel(
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
  TORCH_CHECK(src.is_cuda(), "scatter_min (HIP): src must be a CUDA tensor");
  TORCH_CHECK(index.is_cuda(),
              "scatter_min (HIP): index must be a CUDA tensor");
  TORCH_CHECK(src.device() == index.device(),
              "scatter_min (HIP): src and index must be on the same device");
  TORCH_CHECK(src.dim() == index.dim(),
              "scatter_min (HIP): src.dim() must equal index.dim() after "
              "broadcasting (got src.dim()=",
              src.dim(), ", index.dim()=", index.dim(), ")");

  dim = dim < 0 ? src.dim() + dim : dim;
  TORCH_CHECK(dim >= 0 && dim < src.dim(),
              "scatter_min (HIP): dim out of range");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

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
                    "scatter_min (HIP): out.size(", i,
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
    out = at::empty(sizes, src_c.options());
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
        "scatter_min_hip_init_out",
        [&] { out.fill_(std::numeric_limits<scalar_t>::max()); });
  }

  auto arg_out =
      at::full(out.sizes(), E, index_c.options().dtype(at::ScalarType::Long));

  if (src_c.numel() == 0) {
    if (!out_was_provided) {
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
      "scatter_min_hip", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        const auto* index_data = index_c.data_ptr<int64_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        auto* arg_out_data = arg_out.data_ptr<int64_t>();

        scatter_min_value_hip_kernel<scalar_t>
            <<<blocks((int)numel), threads(), 0, stream>>>(
                src_data, index_data, out_data, E, K, N, numel);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        scatter_min_arg_hip_kernel<scalar_t>
            <<<blocks((int)numel), threads(), 0, stream>>>(
                src_data, index_data, out_data, arg_out_data, E, K, N, numel);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  if (!out_was_provided) {
    out.masked_fill_(arg_out == E, 0);
  }

  return std::make_tuple(out, arg_out);
}

// Value-pass kernel.
template <typename scalar_t>
__global__ void scatter_max_value_hip_kernel(
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

// Arg-pass kernel.
template <typename scalar_t>
__global__ void scatter_max_arg_hip_kernel(
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
  TORCH_CHECK(src.is_cuda(), "scatter_max (HIP): src must be a CUDA tensor");
  TORCH_CHECK(index.is_cuda(),
              "scatter_max (HIP): index must be a CUDA tensor");
  TORCH_CHECK(src.device() == index.device(),
              "scatter_max (HIP): src and index must be on the same device");
  TORCH_CHECK(src.dim() == index.dim(),
              "scatter_max (HIP): src.dim() must equal index.dim() after "
              "broadcasting (got src.dim()=",
              src.dim(), ", index.dim()=", index.dim(), ")");

  dim = dim < 0 ? src.dim() + dim : dim;
  TORCH_CHECK(dim >= 0 && dim < src.dim(),
              "scatter_max (HIP): dim out of range");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

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
                    "scatter_max (HIP): out.size(", i,
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
    out = at::empty(sizes, src_c.options());
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
        "scatter_max_hip_init_out",
        [&] { out.fill_(std::numeric_limits<scalar_t>::lowest()); });
  }

  auto arg_out =
      at::full(out.sizes(), E, index_c.options().dtype(at::ScalarType::Long));

  if (src_c.numel() == 0) {
    if (!out_was_provided) {
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
      "scatter_max_hip", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        const auto* index_data = index_c.data_ptr<int64_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        auto* arg_out_data = arg_out.data_ptr<int64_t>();

        scatter_max_value_hip_kernel<scalar_t>
            <<<blocks((int)numel), threads(), 0, stream>>>(
                src_data, index_data, out_data, E, K, N, numel);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        scatter_max_arg_hip_kernel<scalar_t>
            <<<blocks((int)numel), threads(), 0, stream>>>(
                src_data, index_data, out_data, arg_out_data, E, K, N, numel);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

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
