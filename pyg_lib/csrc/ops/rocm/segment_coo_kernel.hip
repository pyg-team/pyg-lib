#include "../segment_coo.h"

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

#include "atomics.cuh"
#include "shuffle.cuh"

namespace pyg {
namespace ops {

namespace {

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#define FULL_MASK 0xffffffff

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

// K==1 kernel — one thread per `src` entry, warp-reduce equal-index runs.
template <typename scalar_t>
__global__ void segment_sum_coo_hip_kernel(
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
      tmp = SHFL_UP_SYNC(FULL_MASK, val, i);
      next_idx = SHFL_UP_SYNC(FULL_MASK, idx, i);
      if (lane_idx >= i && row_idx / D == (row_idx - i) / D) {
        if (idx == next_idx) {
          val = val + tmp;
        }
      }
    }

    next_idx = SHFL_DOWN_SYNC(FULL_MASK, idx, 1);
    if (lane_idx == WARP_SIZE - 1 || row_idx / D != (row_idx + 1) / D ||
        idx != next_idx) {
      atomicAdd(out_data + out_idx, val);
    }
  }
}

// K>1 broadcast kernel.
template <typename scalar_t, int TB>
__global__ void segment_sum_coo_broadcast_hip_kernel(
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
              "segment_sum_coo (HIP): src must be a CUDA tensor");
  TORCH_CHECK(index.is_cuda(),
              "segment_sum_coo (HIP): index must be a CUDA tensor");
  TORCH_CHECK(src.device() == index.device(),
              "segment_sum_coo (HIP): src and index must be on the same "
              "device");
  TORCH_CHECK(src.dim() >= index.dim(),
              "segment_sum_coo (HIP): src.dim() must be >= index.dim() (got ",
              src.dim(), " vs ", index.dim(), ")");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

  auto sizes = index.sizes().vec();
  for (int64_t i = 0; i < index.dim(); ++i) {
    sizes[i] = src.size(i);
  }
  auto index_b = index.expand(sizes);

  const auto dim = index_b.dim() - 1;

  auto src_c = src.contiguous();
  auto index_c = index_b.contiguous();

  at::Tensor out;
  const bool out_was_provided = optional_out.has_value();
  if (out_was_provided) {
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "segment_sum_coo (HIP): out.size(", i,
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
      "segment_sum_coo_hip", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();

        if (K == 1) {
          const int T = threads();
          const int B = std::max(1, static_cast<int>((E + T - 1) / T));
          segment_sum_coo_hip_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, index_info, out_data, E, N);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          const dim3 block_dim(32, 8);
          const dim3 grid_cols((K + 31) / 32);
          if (avg_len <= 8.0f) {
            constexpr int TB = 4;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_sum_coo_broadcast_hip_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else if (avg_len <= 16.0f) {
            constexpr int TB = 8;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_sum_coo_broadcast_hip_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else if (avg_len <= 32.0f) {
            constexpr int TB = 16;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_sum_coo_broadcast_hip_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else {
            constexpr int TB = 32;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_sum_coo_broadcast_hip_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        }
      });

  return out;
}

__global__ void segment_mean_coo_count_hip_kernel(
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
              "segment_mean_coo (HIP): src must be a CUDA tensor");
  TORCH_CHECK(index.is_cuda(),
              "segment_mean_coo (HIP): index must be a CUDA tensor");
  TORCH_CHECK(src.device() == index.device(),
              "segment_mean_coo (HIP): src and index must be on the same "
              "device");
  TORCH_CHECK(src.dim() >= index.dim(),
              "segment_mean_coo (HIP): src.dim() must be >= index.dim() (got ",
              src.dim(), " vs ", index.dim(), ")");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

  auto sizes = index.sizes().vec();
  for (int64_t i = 0; i < index.dim(); ++i) {
    sizes[i] = src.size(i);
  }
  auto index_b = index.expand(sizes);

  const auto dim = index_b.dim() - 1;

  auto src_c = src.contiguous();
  auto index_c = index_b.contiguous();

  at::Tensor out;
  const bool out_was_provided = optional_out.has_value();
  if (out_was_provided) {
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "segment_mean_coo (HIP): out.size(", i,
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

  auto count_sizes = index_c.sizes().vec();
  count_sizes[dim] = N;
  auto count = at::zeros(count_sizes, index_c.options());

  auto index_info = at::cuda::detail::getTensorInfo<int64_t, int>(index_c);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "segment_mean_coo_hip", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();

        if (K == 1) {
          const int T = threads();
          const int B = std::max(1, static_cast<int>((E + T - 1) / T));
          segment_sum_coo_hip_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, index_info, out_data, E, N);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          const dim3 block_dim(32, 8);
          const dim3 grid_cols((K + 31) / 32);
          if (avg_len <= 8.0f) {
            constexpr int TB = 4;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_sum_coo_broadcast_hip_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else if (avg_len <= 16.0f) {
            constexpr int TB = 8;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_sum_coo_broadcast_hip_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else if (avg_len <= 32.0f) {
            constexpr int TB = 16;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_sum_coo_broadcast_hip_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else {
            constexpr int TB = 32;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_sum_coo_broadcast_hip_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        }
      });

  {
    const int T = threads();
    const int B = std::max(1, static_cast<int>((E + T - 1) / T));
    segment_mean_coo_count_hip_kernel<<<B, T, 0, stream>>>(
        index_info, count.data_ptr<int64_t>(), E, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

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

template <typename scalar_t>
static inline __device__ scalar_t device_min(scalar_t a, scalar_t b) {
  if constexpr (std::is_same_v<scalar_t, at::Half> ||
                std::is_same_v<scalar_t, at::BFloat16>) {
    return static_cast<float>(a) < static_cast<float>(b) ? a : b;
  } else {
    return a < b ? a : b;
  }
}

template <typename scalar_t>
static inline __device__ scalar_t device_max(scalar_t a, scalar_t b) {
  if constexpr (std::is_same_v<scalar_t, at::Half> ||
                std::is_same_v<scalar_t, at::BFloat16>) {
    return static_cast<float>(a) > static_cast<float>(b) ? a : b;
  } else {
    return a > b ? a : b;
  }
}

// Value-pass K==1 kernel for min.
template <typename scalar_t>
__global__ void segment_min_coo_hip_kernel(
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
      tmp = SHFL_UP_SYNC(FULL_MASK, val, i);
      next_idx = SHFL_UP_SYNC(FULL_MASK, idx, i);
      if (lane_idx >= i && row_idx / D == (row_idx - i) / D) {
        if (idx == next_idx) {
          val = device_min(val, tmp);
        }
      }
    }

    next_idx = SHFL_DOWN_SYNC(FULL_MASK, idx, 1);
    if (lane_idx == WARP_SIZE - 1 || row_idx / D != (row_idx + 1) / D ||
        idx != next_idx) {
      atomicMin(out_data + out_idx, val);
    }
  }
}

// Value-pass K>1 broadcast kernel for min.
template <typename scalar_t, int TB>
__global__ void segment_min_coo_broadcast_hip_kernel(
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

// Arg-pass K==1 kernel for min.
template <typename scalar_t>
__global__ void segment_min_coo_arg_hip_kernel(
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

// Arg-pass K>1 broadcast kernel for min.
template <typename scalar_t>
__global__ void segment_min_coo_arg_broadcast_hip_kernel(
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
              "segment_min_coo (HIP): src must be a CUDA tensor");
  TORCH_CHECK(index.is_cuda(),
              "segment_min_coo (HIP): index must be a CUDA tensor");
  TORCH_CHECK(src.device() == index.device(),
              "segment_min_coo (HIP): src and index must be on the same "
              "device");
  TORCH_CHECK(src.dim() >= index.dim(),
              "segment_min_coo (HIP): src.dim() must be >= index.dim() (got ",
              src.dim(), " vs ", index.dim(), ")");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

  auto sizes = index.sizes().vec();
  for (int64_t i = 0; i < index.dim(); ++i) {
    sizes[i] = src.size(i);
  }
  auto index_b = index.expand(sizes);

  const auto dim = index_b.dim() - 1;

  auto src_c = src.contiguous();
  auto index_c = index_b.contiguous();

  const int64_t E_dim = src_c.size(dim);

  at::Tensor out;
  const bool out_was_provided = optional_out.has_value();
  if (out_was_provided) {
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "segment_min_coo (HIP): out.size(", i,
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
    out = at::empty(out_sizes, src_c.options());
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
        "segment_min_coo_hip_init_out",
        [&] { out.fill_(std::numeric_limits<scalar_t>::max()); });
  }

  auto arg_out = at::full(out.sizes(), E_dim,
                          index_c.options().dtype(at::ScalarType::Long));

  if (index_c.numel() == 0) {
    if (!out_was_provided) {
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
      "segment_min_coo_hip", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        auto* arg_out_data = arg_out.data_ptr<int64_t>();

        if (K == 1) {
          const int T = threads();
          const int B = std::max(1, static_cast<int>((E + T - 1) / T));
          segment_min_coo_hip_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, index_info, out_data, E, N);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          const dim3 block_dim(32, 8);
          const dim3 grid_cols((K + 31) / 32);
          if (avg_len <= 8.0f) {
            constexpr int TB = 4;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_min_coo_broadcast_hip_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else if (avg_len <= 16.0f) {
            constexpr int TB = 8;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_min_coo_broadcast_hip_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else if (avg_len <= 32.0f) {
            constexpr int TB = 16;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_min_coo_broadcast_hip_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else {
            constexpr int TB = 32;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_min_coo_broadcast_hip_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        }

        if (K == 1) {
          const int T = threads();
          const int B = std::max(1, static_cast<int>((E + T - 1) / T));
          segment_min_coo_arg_hip_kernel<scalar_t><<<B, T, 0, stream>>>(
              src_data, index_info, out_data, arg_out_data, E, N);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          const int T = threads();
          const int B = std::max(1, static_cast<int>((E * K + T - 1) / T));
          segment_min_coo_arg_broadcast_hip_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, index_info, out_data,
                                    arg_out_data, E, K, N);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });

  if (!out_was_provided) {
    out.masked_fill_(arg_out == E_dim, 0);
  }

  return std::make_tuple(out, arg_out);
}

// Value-pass K==1 kernel for max.
template <typename scalar_t>
__global__ void segment_max_coo_hip_kernel(
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
      tmp = SHFL_UP_SYNC(FULL_MASK, val, i);
      next_idx = SHFL_UP_SYNC(FULL_MASK, idx, i);
      if (lane_idx >= i && row_idx / D == (row_idx - i) / D) {
        if (idx == next_idx) {
          val = device_max(val, tmp);
        }
      }
    }

    next_idx = SHFL_DOWN_SYNC(FULL_MASK, idx, 1);
    if (lane_idx == WARP_SIZE - 1 || row_idx / D != (row_idx + 1) / D ||
        idx != next_idx) {
      atomicMax(out_data + out_idx, val);
    }
  }
}

// Value-pass K>1 broadcast kernel for max.
template <typename scalar_t, int TB>
__global__ void segment_max_coo_broadcast_hip_kernel(
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

// Arg-pass K==1 kernel for max.
template <typename scalar_t>
__global__ void segment_max_coo_arg_hip_kernel(
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

// Arg-pass K>1 broadcast kernel for max.
template <typename scalar_t>
__global__ void segment_max_coo_arg_broadcast_hip_kernel(
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
              "segment_max_coo (HIP): src must be a CUDA tensor");
  TORCH_CHECK(index.is_cuda(),
              "segment_max_coo (HIP): index must be a CUDA tensor");
  TORCH_CHECK(src.device() == index.device(),
              "segment_max_coo (HIP): src and index must be on the same "
              "device");
  TORCH_CHECK(src.dim() >= index.dim(),
              "segment_max_coo (HIP): src.dim() must be >= index.dim() (got ",
              src.dim(), " vs ", index.dim(), ")");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

  auto sizes = index.sizes().vec();
  for (int64_t i = 0; i < index.dim(); ++i) {
    sizes[i] = src.size(i);
  }
  auto index_b = index.expand(sizes);

  const auto dim = index_b.dim() - 1;

  auto src_c = src.contiguous();
  auto index_c = index_b.contiguous();

  const int64_t E_dim = src_c.size(dim);

  at::Tensor out;
  const bool out_was_provided = optional_out.has_value();
  if (out_was_provided) {
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "segment_max_coo (HIP): out.size(", i,
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
    out = at::empty(out_sizes, src_c.options());
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
        "segment_max_coo_hip_init_out",
        [&] { out.fill_(std::numeric_limits<scalar_t>::lowest()); });
  }

  auto arg_out = at::full(out.sizes(), E_dim,
                          index_c.options().dtype(at::ScalarType::Long));

  if (index_c.numel() == 0) {
    if (!out_was_provided) {
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
      "segment_max_coo_hip", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        auto* arg_out_data = arg_out.data_ptr<int64_t>();

        if (K == 1) {
          const int T = threads();
          const int B = std::max(1, static_cast<int>((E + T - 1) / T));
          segment_max_coo_hip_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, index_info, out_data, E, N);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          const dim3 block_dim(32, 8);
          const dim3 grid_cols((K + 31) / 32);
          if (avg_len <= 8.0f) {
            constexpr int TB = 4;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_max_coo_broadcast_hip_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else if (avg_len <= 16.0f) {
            constexpr int TB = 8;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_max_coo_broadcast_hip_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else if (avg_len <= 32.0f) {
            constexpr int TB = 16;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_max_coo_broadcast_hip_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else {
            constexpr int TB = 32;
            const dim3 grid_dim((E_1 * ((E_2 + TB - 1) / TB) + 7) / 8,
                                grid_cols.x);
            segment_max_coo_broadcast_hip_kernel<scalar_t, TB>
                <<<grid_dim, block_dim, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        }

        if (K == 1) {
          const int T = threads();
          const int B = std::max(1, static_cast<int>((E + T - 1) / T));
          segment_max_coo_arg_hip_kernel<scalar_t><<<B, T, 0, stream>>>(
              src_data, index_info, out_data, arg_out_data, E, N);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          const int T = threads();
          const int B = std::max(1, static_cast<int>((E * K + T - 1) / T));
          segment_max_coo_arg_broadcast_hip_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, index_info, out_data,
                                    arg_out_data, E, K, N);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });

  if (!out_was_provided) {
    out.masked_fill_(arg_out == E_dim, 0);
  }

  return std::make_tuple(out, arg_out);
}

// gather_coo K==1 kernel.
template <typename scalar_t>
__global__ void gather_coo_hip_kernel(
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

    offset =
        (row_idx / index_info.sizes[index_info.dims - 1]) * static_cast<int>(N);
    out_data[row_idx] = src_data[offset + row];
  }
}

// gather_coo K>1 broadcast kernel.
template <typename scalar_t>
__global__ void gather_coo_broadcast_hip_kernel(
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
  TORCH_CHECK(src.is_cuda(), "gather_coo (HIP): src must be a CUDA tensor");
  TORCH_CHECK(index.is_cuda(),
              "gather_coo (HIP): index must be a CUDA tensor");
  TORCH_CHECK(src.device() == index.device(),
              "gather_coo (HIP): src and index must be on the same device");
  TORCH_CHECK(src.dim() >= index.dim(),
              "gather_coo (HIP): src.dim() must be >= index.dim() (got ",
              src.dim(), " vs ", index.dim(), ")");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

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
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < src_c.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "gather_coo (HIP): out.size(", i, ") must match src.size(",
                    i, ")");
      }
    }
    TORCH_CHECK(index_c.size(dim) == out.size(dim),
                "gather_coo (HIP): out.size(", dim, ") must match index.size(",
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
      "gather_coo_hip", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();

        if (K == 1) {
          const int T = threads();
          const int B = blocks(static_cast<int>(E));
          gather_coo_hip_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, index_info, out_data, E, N);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          const int T = threads();
          const int B = blocks(static_cast<int>(E * K));
          gather_coo_broadcast_hip_kernel<scalar_t>
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
