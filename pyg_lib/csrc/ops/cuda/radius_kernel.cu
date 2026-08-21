#include "../radius.h"
#include "utils.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

namespace {

#define RADIUS_THREADS 256

template <typename scalar_t>
__global__ void radius_cuda_kernel(const scalar_t* __restrict__ x,
                                   const scalar_t* __restrict__ y,
                                   const int64_t* __restrict__ ptr_x,
                                   const int64_t* __restrict__ ptr_y,
                                   int64_t* __restrict__ row,
                                   int64_t* __restrict__ col,
                                   const scalar_t r,
                                   const int64_t n,
                                   const int64_t m,
                                   const int64_t dim,
                                   const int64_t num_examples,
                                   const int64_t max_num_neighbors,
                                   const bool ignore_same_index) {
  const int64_t n_y = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_y >= m)
    return;

  int64_t count = 0;
  const int64_t example_idx = get_example_idx(n_y, ptr_y, num_examples);

  for (int64_t n_x = ptr_x[example_idx]; n_x < ptr_x[example_idx + 1]; n_x++) {
    scalar_t dist = 0;
    for (int64_t d = 0; d < dim; d++) {
      scalar_t diff = x[n_x * dim + d] - y[n_y * dim + d];
      dist += diff * diff;
    }

    if (scalar_lt(dist, r) && !(ignore_same_index && n_y == n_x)) {
      row[n_y * max_num_neighbors + count] = n_y;
      col[n_y * max_num_neighbors + count] = n_x;
      count++;
    }

    if (count >= max_num_neighbors)
      break;
  }
}

at::Tensor radius_cuda(const at::Tensor& x,
                       const at::Tensor& y,
                       const std::optional<at::Tensor>& ptr_x,
                       const std::optional<at::Tensor>& ptr_y,
                       double r,
                       int64_t max_num_neighbors,
                       int64_t num_workers,
                       bool ignore_same_index) {
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), "Inputs must be CUDA tensors");
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous(),
              "Inputs must be contiguous");

  std::optional<at::Tensor> ptr_x_v = ptr_x;
  std::optional<at::Tensor> ptr_y_v = ptr_y;

  if (!ptr_x_v.has_value())
    ptr_x_v =
        at::arange(0, x.size(0) + 1, x.size(0), x.options().dtype(at::kLong));
  if (!ptr_y_v.has_value())
    ptr_y_v =
        at::arange(0, y.size(0) + 1, y.size(0), y.options().dtype(at::kLong));

  TORCH_CHECK(ptr_x_v.value().numel() == ptr_y_v.value().numel(),
              "ptr_x and ptr_y must have the same number of elements");

  auto row =
      at::full({y.size(0) * max_num_neighbors}, -1, ptr_y_v.value().options());
  auto col =
      at::full({y.size(0) * max_num_neighbors}, -1, ptr_y_v.value().options());

  dim3 BLOCKS((y.size(0) + RADIUS_THREADS - 1) / RADIUS_THREADS);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "radius_cuda", [&] {
        radius_cuda_kernel<scalar_t><<<BLOCKS, RADIUS_THREADS, 0, stream>>>(
            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
            ptr_x_v.value().data_ptr<int64_t>(),
            ptr_y_v.value().data_ptr<int64_t>(), row.data_ptr<int64_t>(),
            col.data_ptr<int64_t>(), (scalar_t)(r * r), x.size(0), y.size(0),
            x.size(1), ptr_x_v.value().numel() - 1, max_num_neighbors,
            ignore_same_index);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  auto mask = row != -1;
  return at::stack({row.masked_select(mask), col.masked_select(mask)}, 0);
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::radius"), TORCH_FN(radius_cuda));
}

}  // namespace ops
}  // namespace pyg
