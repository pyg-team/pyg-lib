#include "../segment_csr.h"

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

#include "shuffle.cuh"

namespace pyg {
namespace ops {

namespace {

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef FULL_MASK
#define FULL_MASK 0xffffffff
#endif

template <typename scalar_t>
static inline __device__ bool device_lt(scalar_t a, scalar_t b) {
  if constexpr (std::is_same_v<scalar_t, at::Half> ||
                std::is_same_v<scalar_t, at::BFloat16>) {
    return static_cast<float>(a) < static_cast<float>(b);
  } else {
    return a < b;
  }
}

template <typename scalar_t>
static inline __device__ bool device_gt(scalar_t a, scalar_t b) {
  if constexpr (std::is_same_v<scalar_t, at::Half> ||
                std::is_same_v<scalar_t, at::BFloat16>) {
    return static_cast<float>(a) > static_cast<float>(b);
  } else {
    return a > b;
  }
}

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

// Strided-offset helper specialised for CSR `indptr` row pointers.
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

// K==1 kernel — one thread per row.
template <typename scalar_t>
__global__ void segment_sum_csr_hip_kernel(
    const scalar_t* __restrict__ src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> indptr_info,
    scalar_t* __restrict__ out_data,
    size_t N,
    size_t E) {
  using opmath_t = at::opmath_type<scalar_t>;

  int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_idx >= static_cast<int>(N))
    return;

  int offset = IndexPtrToOffset::get(row_idx, indptr_info);
  int64_t row_start = __ldg(indptr_info.data + offset);
  int64_t row_end = __ldg(indptr_info.data + offset +
                          indptr_info.strides[indptr_info.dims - 1]);

  int batch_offset = (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) *
                     static_cast<int>(E);

  opmath_t val = static_cast<opmath_t>(0);
  for (int64_t src_idx = row_start; src_idx < row_end; ++src_idx) {
    val += static_cast<opmath_t>(src_data[batch_offset + src_idx]);
  }

  out_data[row_idx] =
      static_cast<scalar_t>(static_cast<opmath_t>(out_data[row_idx]) + val);
}

// K>1 broadcast kernel — one thread per (row, col) pair.
template <typename scalar_t>
__global__ void segment_sum_csr_broadcast_hip_kernel(
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

  int batch_offset = (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) *
                     static_cast<int>(E) * static_cast<int>(K);

  opmath_t val = static_cast<opmath_t>(0);
  for (int64_t src_idx = row_start; src_idx < row_end; ++src_idx) {
    val += static_cast<opmath_t>(
        src_data[batch_offset + static_cast<int>(K) * src_idx + col_idx]);
  }

  out_data[thread_idx] =
      static_cast<scalar_t>(static_cast<opmath_t>(out_data[thread_idx]) + val);
}

at::Tensor segment_sum_csr_kernel(
    const at::Tensor& src,
    const at::Tensor& indptr,
    const std::optional<at::Tensor>& optional_out) {
  TORCH_CHECK(src.is_cuda(),
              "segment_sum_csr (HIP): src must be a CUDA tensor");
  TORCH_CHECK(indptr.is_cuda(),
              "segment_sum_csr (HIP): indptr must be a CUDA tensor");
  TORCH_CHECK(src.device() == indptr.device(),
              "segment_sum_csr (HIP): src and indptr must be on the same "
              "device");
  TORCH_CHECK(src.dim() >= indptr.dim(),
              "segment_sum_csr (HIP): src.dim() must be >= indptr.dim() (got ",
              src.dim(), " vs ", indptr.dim(), ")");
  TORCH_CHECK(indptr.dim() >= 1,
              "segment_sum_csr (HIP): indptr must have at least 1 dimension");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

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
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "segment_sum_csr (HIP): out.size(", i,
                    ") must match src.size(", i, ")");
      }
    }
    TORCH_CHECK(src_c.numel() == 0 || out.size(dim) == indptr_c.size(dim) - 1,
                "segment_sum_csr (HIP): out.size(", dim,
                ") must equal indptr.size(-1) - 1 (got ", out.size(dim), " vs ",
                indptr_c.size(dim) - 1, ")");
  } else {
    auto out_sizes = src_c.sizes().vec();
    out_sizes[dim] = std::max<int64_t>(indptr_c.size(dim) - 1, 0);
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

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "segment_sum_csr_hip", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();

        if (K == 1) {
          const int T = threads();
          const int B = blocks(static_cast<int>(N));
          segment_sum_csr_hip_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, indptr_info, out_data, N, E);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          const int T = threads();
          const int B = blocks(static_cast<int>(N * K));
          segment_sum_csr_broadcast_hip_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, indptr_info, out_data, N, K, E);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });

  return out;
}

at::Tensor segment_mean_csr_kernel(
    const at::Tensor& src,
    const at::Tensor& indptr,
    const std::optional<at::Tensor>& optional_out) {
  TORCH_CHECK(src.is_cuda(),
              "segment_mean_csr (HIP): src must be a CUDA tensor");
  TORCH_CHECK(indptr.is_cuda(),
              "segment_mean_csr (HIP): indptr must be a CUDA tensor");
  TORCH_CHECK(src.device() == indptr.device(),
              "segment_mean_csr (HIP): src and indptr must be on the same "
              "device");
  TORCH_CHECK(src.dim() >= indptr.dim(),
              "segment_mean_csr (HIP): src.dim() must be >= indptr.dim() "
              "(got ",
              src.dim(), " vs ", indptr.dim(), ")");
  TORCH_CHECK(indptr.dim() >= 1,
              "segment_mean_csr (HIP): indptr must have at least 1 dimension");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

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
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "segment_mean_csr (HIP): out.size(", i,
                    ") must match src.size(", i, ")");
      }
    }
    TORCH_CHECK(src_c.numel() == 0 || out.size(dim) == indptr_c.size(dim) - 1,
                "segment_mean_csr (HIP): out.size(", dim,
                ") must equal indptr.size(-1) - 1 (got ", out.size(dim), " vs ",
                indptr_c.size(dim) - 1, ")");
  } else {
    auto out_sizes = src_c.sizes().vec();
    out_sizes[dim] = std::max<int64_t>(indptr_c.size(dim) - 1, 0);
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

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "segment_mean_csr_hip", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();

        if (K == 1) {
          const int T = threads();
          const int B = blocks(static_cast<int>(N));
          segment_sum_csr_hip_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, indptr_info, out_data, N, E);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          const int T = threads();
          const int B = blocks(static_cast<int>(N * K));
          segment_sum_csr_broadcast_hip_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, indptr_info, out_data, N, K, E);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });

  auto count = indptr_c.diff(/*n=*/1, /*dim=*/-1).to(out.options());
  count.masked_fill_(count == 0, 1);
  for (int64_t i = dim + 1; i < out.dim(); ++i) {
    count = count.unsqueeze(-1);
  }
  out.true_divide_(count);

  return out;
}

// gather_csr K==1 kernel.
template <typename scalar_t>
__global__ void gather_csr_hip_kernel(
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
  scalar_t val = src_data[row_idx];

  int batch_offset = (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) *
                     static_cast<int>(E);
  for (int out_idx = row_start; out_idx < row_end; ++out_idx) {
    out_data[batch_offset + out_idx] = val;
  }
}

// gather_csr K>1 broadcast kernel.
template <typename scalar_t>
__global__ void gather_csr_broadcast_hip_kernel(
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
  TORCH_CHECK(src.is_cuda(), "gather_csr (HIP): src must be a CUDA tensor");
  TORCH_CHECK(indptr.is_cuda(),
              "gather_csr (HIP): indptr must be a CUDA tensor");
  TORCH_CHECK(src.device() == indptr.device(),
              "gather_csr (HIP): src and indptr must be on the same device");
  TORCH_CHECK(src.dim() >= indptr.dim(),
              "gather_csr (HIP): src.dim() must be >= indptr.dim() (got ",
              src.dim(), " vs ", indptr.dim(), ")");
  TORCH_CHECK(indptr.dim() >= 1,
              "gather_csr (HIP): indptr must have at least 1 dimension");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

  auto sizes = indptr.sizes().vec();
  for (int64_t i = 0; i < indptr.dim() - 1; ++i) {
    sizes[i] = src.size(i);
  }
  auto indptr_b = indptr.expand(sizes);

  const auto dim = indptr_b.dim() - 1;
  TORCH_CHECK(src.size(dim) == 0 || src.size(dim) == indptr_b.size(dim) - 1,
              "gather_csr (HIP): src.size(", dim,
              ") must be 0 or equal to indptr.size(-1) - 1 (got ",
              src.size(dim), " vs ", indptr_b.size(dim) - 1, ")");

  auto src_c = src.contiguous();
  auto indptr_c = indptr_b.contiguous();

  at::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "gather_csr (HIP): out.size(", i, ") must match src.size(",
                    i, ")");
      }
    }
  } else {
    auto out_sizes = src_c.sizes().vec();
    if (src_c.numel() > 0) {
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

  const auto N = src_c.size(dim) * (indptr_c.numel() / indptr_c.size(-1));
  const auto K = src_c.numel() / N;
  const auto E = out.size(dim);

  auto indptr_info = at::cuda::detail::getTensorInfo<int64_t, int>(indptr_c);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "gather_csr_hip", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();

        if (K == 1) {
          const int T = threads();
          const int B = blocks(static_cast<int>(N));
          gather_csr_hip_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, indptr_info, out_data, N, E);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          const int T = threads();
          const int B = blocks(static_cast<int>(N * K));
          gather_csr_broadcast_hip_kernel<scalar_t>
              <<<B, T, 0, stream>>>(src_data, indptr_info, out_data, N, K, E);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });

  return out;
}

// K==1 min kernel — one warp per row.
template <typename scalar_t>
__global__ void segment_min_csr_hip_kernel(
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

  scalar_t val = std::numeric_limits<scalar_t>::max();
  int64_t arg = static_cast<int64_t>(E);

  for (int64_t src_idx = row_start + lane_idx; src_idx < row_end;
       src_idx += TB) {
    scalar_t cand = src_data[batch_offset + src_idx];
    if (device_lt(cand, val)) {
      val = cand;
      arg = src_idx;
    }
  }

#pragma unroll
  for (int i = TB / 2; i > 0; i /= 2) {
    scalar_t tmp_val = SHFL_DOWN_SYNC(FULL_MASK, val, i);
    int64_t tmp_arg = SHFL_DOWN_SYNC(FULL_MASK, arg, i);
    if (device_lt(tmp_val, val)) {
      val = tmp_val;
      arg = tmp_arg;
    }
  }

  if (lane_idx == 0) {
    out_data[row_idx] = val;
    arg_out_data[row_idx] = arg;
  }
}

// K>1 broadcast min kernel.
template <typename scalar_t>
__global__ void segment_min_csr_broadcast_hip_kernel(
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
              "segment_min_csr (HIP): src must be a CUDA tensor");
  TORCH_CHECK(indptr.is_cuda(),
              "segment_min_csr (HIP): indptr must be a CUDA tensor");
  TORCH_CHECK(src.device() == indptr.device(),
              "segment_min_csr (HIP): src and indptr must be on the same "
              "device");
  TORCH_CHECK(src.dim() >= indptr.dim(),
              "segment_min_csr (HIP): src.dim() must be >= indptr.dim() (got ",
              src.dim(), " vs ", indptr.dim(), ")");
  TORCH_CHECK(indptr.dim() >= 1,
              "segment_min_csr (HIP): indptr must have at least 1 dimension");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

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
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "segment_min_csr (HIP): out.size(", i,
                    ") must match src.size(", i, ")");
      }
    }
    TORCH_CHECK(src_c.numel() == 0 || out.size(dim) == indptr_c.size(dim) - 1,
                "segment_min_csr (HIP): out.size(", dim,
                ") must equal indptr.size(-1) - 1 (got ", out.size(dim), " vs ",
                indptr_c.size(dim) - 1, ")");
  } else {
    auto out_sizes = src_c.sizes().vec();
    out_sizes[dim] = std::max<int64_t>(indptr_c.size(dim) - 1, 0);
    out = at::empty(out_sizes, src_c.options());
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
        "segment_min_csr_hip_init_out",
        [&] { out.fill_(std::numeric_limits<scalar_t>::max()); });
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
      "segment_min_csr_hip", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        auto* arg_out_data = arg_out.data_ptr<int64_t>();

        if (K == 1) {
          constexpr int TB = WARP_SIZE;
          const int T = threads();
          const int B =
              std::max<int>(1, static_cast<int>((TB * N + T - 1) / T));
          segment_min_csr_hip_kernel<scalar_t><<<B, T, 0, stream>>>(
              src_data, indptr_info, out_data, arg_out_data, N, E);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          const int T = threads();
          const int B = blocks(static_cast<int>(N * K));
          segment_min_csr_broadcast_hip_kernel<scalar_t><<<B, T, 0, stream>>>(
              src_data, indptr_info, out_data, arg_out_data, N, K, E);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });

  if (!out_was_provided) {
    out.masked_fill_(arg_out == E_dim, 0);
  }

  return std::make_tuple(out, arg_out);
}

// K==1 max kernel — one warp per row.
template <typename scalar_t>
__global__ void segment_max_csr_hip_kernel(
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

  scalar_t val = std::numeric_limits<scalar_t>::lowest();
  int64_t arg = static_cast<int64_t>(E);

  for (int64_t src_idx = row_start + lane_idx; src_idx < row_end;
       src_idx += TB) {
    scalar_t cand = src_data[batch_offset + src_idx];
    if (device_gt(cand, val)) {
      val = cand;
      arg = src_idx;
    }
  }

#pragma unroll
  for (int i = TB / 2; i > 0; i /= 2) {
    scalar_t tmp_val = SHFL_DOWN_SYNC(FULL_MASK, val, i);
    int64_t tmp_arg = SHFL_DOWN_SYNC(FULL_MASK, arg, i);
    if (device_gt(tmp_val, val)) {
      val = tmp_val;
      arg = tmp_arg;
    }
  }

  if (lane_idx == 0) {
    out_data[row_idx] = val;
    arg_out_data[row_idx] = arg;
  }
}

// K>1 broadcast max kernel.
template <typename scalar_t>
__global__ void segment_max_csr_broadcast_hip_kernel(
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
              "segment_max_csr (HIP): src must be a CUDA tensor");
  TORCH_CHECK(indptr.is_cuda(),
              "segment_max_csr (HIP): indptr must be a CUDA tensor");
  TORCH_CHECK(src.device() == indptr.device(),
              "segment_max_csr (HIP): src and indptr must be on the same "
              "device");
  TORCH_CHECK(src.dim() >= indptr.dim(),
              "segment_max_csr (HIP): src.dim() must be >= indptr.dim() (got ",
              src.dim(), " vs ", indptr.dim(), ")");
  TORCH_CHECK(indptr.dim() >= 1,
              "segment_max_csr (HIP): indptr must have at least 1 dimension");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(src));

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
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim) {
        TORCH_CHECK(src_c.size(i) == out.size(i),
                    "segment_max_csr (HIP): out.size(", i,
                    ") must match src.size(", i, ")");
      }
    }
    TORCH_CHECK(src_c.numel() == 0 || out.size(dim) == indptr_c.size(dim) - 1,
                "segment_max_csr (HIP): out.size(", dim,
                ") must equal indptr.size(-1) - 1 (got ", out.size(dim), " vs ",
                indptr_c.size(dim) - 1, ")");
  } else {
    auto out_sizes = src_c.sizes().vec();
    out_sizes[dim] = std::max<int64_t>(indptr_c.size(dim) - 1, 0);
    out = at::empty(out_sizes, src_c.options());
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
        "segment_max_csr_hip_init_out",
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
      "segment_max_csr_hip", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        auto* arg_out_data = arg_out.data_ptr<int64_t>();

        if (K == 1) {
          constexpr int TB = WARP_SIZE;
          const int T = threads();
          const int B =
              std::max<int>(1, static_cast<int>((TB * N + T - 1) / T));
          segment_max_csr_hip_kernel<scalar_t><<<B, T, 0, stream>>>(
              src_data, indptr_info, out_data, arg_out_data, N, E);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          const int T = threads();
          const int B = blocks(static_cast<int>(N * K));
          segment_max_csr_broadcast_hip_kernel<scalar_t><<<B, T, 0, stream>>>(
              src_data, indptr_info, out_data, arg_out_data, N, K, E);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });

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
