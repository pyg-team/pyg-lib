#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

namespace {

template <typename scalar_t>
void scatter_add_cpu_kernel(const at::Tensor& src,
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
  
  auto src_data = src.data_ptr<scalar_t>();
  auto index_data = index.data_ptr<int64_t>();
  auto out_data = out.data_ptr<scalar_t>();
  
  // Parallel processing over batches
  at::parallel_for(0, B, 1, [&](int64_t b_start, int64_t b_end) {
    for (int64_t b = b_start; b < b_end; b++) {
      for (int64_t e = 0; e < E; e++) {
        for (int64_t k = 0; k < K; k++) {
          int64_t src_idx = b * E * K + e * K + k;
          
          // Calculate index position for this element
          // Index tensor is broadcasted to match src dimensions
          int64_t index_idx = src_idx;
          int64_t scatter_idx = index_data[index_idx];
          
          // Bounds check
          if (scatter_idx >= 0 && scatter_idx < N) {
            int64_t out_idx = b * N * K + scatter_idx * K + k;
            out_data[out_idx] += src_data[src_idx];
          }
        }
      }
    }
  });
}

}  // namespace

at::Tensor scatter_add_kernel_cpu_impl(const at::Tensor& src,
                                       const at::Tensor& index,
                                       const int64_t dim,
                                       at::Tensor out) {
  TORCH_CHECK(src.is_cpu(), "Input tensor must be CPU tensor");
  TORCH_CHECK(index.is_cpu(), "Index tensor must be CPU tensor");
  TORCH_CHECK(out.is_cpu(), "Output tensor must be CPU tensor");
  
  // For simple cases, use PyTorch's optimized scatter_add_
  if (src.is_contiguous() && index.is_contiguous() && out.is_contiguous()) {
    return out.scatter_add_(dim, index, src);
  }
  
  // For complex broadcasting cases, use our custom kernel
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                              src.scalar_type(), "scatter_add_cpu", [&] {
    scatter_add_cpu_kernel<scalar_t>(src, index, dim, out);
  });
  
  return out;
}

}  // namespace ops
}  // namespace pyg