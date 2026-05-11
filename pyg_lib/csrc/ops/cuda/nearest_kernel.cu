#include "../nearest.h"
#include "utils.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

namespace {

#define NEAREST_THREADS 1024

template <typename scalar_t>
__global__ void nearest_cuda_kernel(const scalar_t* __restrict__ x,
                                    const scalar_t* __restrict__ y,
                                    const int64_t* __restrict__ ptr_x,
                                    const int64_t* __restrict__ ptr_y,
                                    int64_t* __restrict__ out,
                                    int64_t batch_size,
                                    int64_t dim) {
  const int64_t thread_idx = threadIdx.x;
  const int64_t n_x = blockIdx.x;

  int64_t batch_idx = 0;
  for (int64_t b = 0; b < batch_size; b++) {
    if (n_x >= ptr_x[b] && n_x < ptr_x[b + 1]) {
      batch_idx = b;
      break;
    }
  }

  const int64_t y_start = ptr_y[batch_idx];
  const int64_t y_end = ptr_y[batch_idx + 1];

  __shared__ scalar_t best_dist[NEAREST_THREADS];
  __shared__ int64_t best_dist_idx[NEAREST_THREADS];

  scalar_t best = (scalar_t)1e38;
  int64_t best_idx = y_start;
  for (int64_t n_y = y_start + thread_idx; n_y < y_end;
       n_y += NEAREST_THREADS) {
    scalar_t dist = 0;
    for (int64_t d = 0; d < dim; d++) {
      scalar_t diff = x[n_x * dim + d] - y[n_y * dim + d];
      dist += diff * diff;
    }

    if (scalar_lt(dist, best)) {
      best = dist;
      best_idx = n_y;
    }
  }

  best_dist[thread_idx] = best;
  best_dist_idx[thread_idx] = best_idx;

  for (int64_t u = 0; (1 << u) < NEAREST_THREADS; u++) {
    __syncthreads();
    if (thread_idx < (NEAREST_THREADS >> (u + 1))) {
      int64_t idx_1 = (thread_idx * 2) << u;
      int64_t idx_2 = (thread_idx * 2 + 1) << u;
      if (scalar_gt(best_dist[idx_1], best_dist[idx_2])) {
        best_dist[idx_1] = best_dist[idx_2];
        best_dist_idx[idx_1] = best_dist_idx[idx_2];
      }
    }
  }

  __syncthreads();
  if (thread_idx == 0) {
    out[n_x] = best_dist_idx[0];
  }
}

at::Tensor nearest_cuda(const at::Tensor& x,
                        const at::Tensor& y,
                        const std::optional<at::Tensor>& ptr_x,
                        const std::optional<at::Tensor>& ptr_y) {
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

  auto out = at::empty({x.size(0)}, ptr_x_v.value().options());

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "nearest_cuda", [&] {
    nearest_cuda_kernel<scalar_t><<<x.size(0), NEAREST_THREADS, 0, stream>>>(
        x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
        ptr_x_v.value().data_ptr<int64_t>(),
        ptr_y_v.value().data_ptr<int64_t>(), out.data_ptr<int64_t>(),
        ptr_x_v.value().size(0) - 1, x.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::nearest"), TORCH_FN(nearest_cuda));
}

}  // namespace ops
}  // namespace pyg
