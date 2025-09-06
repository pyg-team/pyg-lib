#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <torch/library.h>

namespace pyg {
namespace ops {

namespace {

#define THREADS 256
#define BLOCKS(N) (((N) + THREADS - 1) / THREADS)

template <typename scalar_t>
__global__ void scatter_add_kernel_impl(
    const scalar_t* __restrict__ src_data,
    const int64_t* __restrict__ index_data,
    scalar_t* __restrict__ out_data,
    const at::cuda::detail::TensorInfo<int64_t, int> index_info,
    const int64_t B,  // Batch size (dimensions before scatter dim)
    const int64_t E,  // Input size along scatter dimension
    const int64_t K,  // Feature size (dimensions after scatter dim)
    const int64_t N,  // Output size along scatter dimension
    const int64_t numel) {
  
  int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (thread_idx >= numel)
    return;
  
  // Calculate indices
  int64_t b = thread_idx / (E * K);  // Batch index
  int64_t e = (thread_idx / K) % E;  // Element index within scatter dimension
  int64_t k = thread_idx % K;        // Feature index
  
  // Get the scatter index for this element
  int64_t index_offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
      thread_idx, index_info);
  int64_t scatter_idx = index_data[index_offset];
  
  // Bounds check
  if (scatter_idx >= 0 && scatter_idx < N) {
    // Compute output position
    int64_t out_pos = b * N * K + scatter_idx * K + k;
    
    // Atomic add to handle race conditions
    atomicAdd(&out_data[out_pos], src_data[thread_idx]);
  }
}

template <typename scalar_t>
void scatter_add_cuda_kernel(const at::Tensor& src,
                             const at::Tensor& index, 
                             const int64_t dim,
                             at::Tensor& out) {
  
  // Calculate tensor dimensions
  int64_t B = 1;  // Batch size (dimensions before scatter dim)
  for (int64_t i = 0; i < dim; i++) {
    B *= src.size(i);
  }
  
  int64_t E = src.size(dim);    // Input size along scatter dimension
  int64_t N = out.size(dim);    // Output size along scatter dimension
  
  int64_t K = 1;  // Feature size (dimensions after scatter dim)
  for (int64_t i = dim + 1; i < src.dim(); i++) {
    K *= src.size(i);
  }
  
  int64_t numel = src.numel();
  
  if (numel == 0) return;
  
  // Get tensor info for index
  auto index_info = at::cuda::detail::getTensorInfo<int64_t, int>(index);
  
  // Launch kernel
  dim3 blocks(BLOCKS(numel));
  dim3 threads(THREADS);
  
  auto stream = at::cuda::getCurrentCUDAStream();
  scatter_add_kernel_impl<scalar_t><<<blocks, threads, 0, stream>>>(
      src.data_ptr<scalar_t>(),
      index.data_ptr<int64_t>(),
      out.data_ptr<scalar_t>(),
      index_info,
      B, E, K, N, numel);
  
  AT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

at::Tensor scatter_add_kernel_cuda(const at::Tensor& src,
                                   const at::Tensor& index,
                                   const int64_t dim,
                                   at::Tensor out) {
  TORCH_CHECK(src.is_cuda(), "Input tensor must be CUDA tensor");
  TORCH_CHECK(index.is_cuda(), "Index tensor must be CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "Output tensor must be CUDA tensor");
  
  // Dispatch to appropriate kernel based on data type
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                              src.scalar_type(), "scatter_add_cuda", [&] {
    scatter_add_cuda_kernel<scalar_t>(src, index, dim, out);
  });
  
  return out;
}

at::Tensor scatter_add_kernel_cpu(const at::Tensor& src,
                                  const at::Tensor& index,
                                  const int64_t dim,
                                  at::Tensor out) {
  // For CPU, fall back to PyTorch's built-in scatter_add_
  // This is already optimized for CPU
  return out.scatter_add_(dim, index, src);
}

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl("scatter_add_kernel", scatter_add_kernel_cuda);
}

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl("scatter_add_kernel", scatter_add_kernel_cpu);
}

// Register kernel schema
TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def("scatter_add_kernel(Tensor src, Tensor index, int dim, Tensor out) -> Tensor");
}

}  // namespace ops
}  // namespace pyg