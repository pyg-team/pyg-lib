#include "../knn.h"
#include "utils.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

namespace {

#define KNN_THREADS 256

template <typename scalar_t>
struct Cosine {
  static inline __device__ scalar_t dot(const scalar_t* a,
                                        const scalar_t* b,
                                        int64_t n_a,
                                        int64_t n_b,
                                        int64_t size) {
    scalar_t result = 0;
    for (int64_t i = 0; i < size; i++) {
      result += a[n_a * size + i] * b[n_b * size + i];
    }
    return result;
  }

  static inline __device__ scalar_t norm(const scalar_t* a,
                                         int64_t n_a,
                                         int64_t size) {
    scalar_t result = 0;
    for (int64_t i = 0; i < size; i++) {
      result += a[n_a * size + i] * a[n_a * size + i];
    }
    return sqrt(result);
  }
};

template <typename scalar_t>
__global__ void knn_cuda_kernel(const scalar_t* __restrict__ x,
                                const scalar_t* __restrict__ y,
                                const int64_t* __restrict__ ptr_x,
                                const int64_t* __restrict__ ptr_y,
                                int64_t* __restrict__ row,
                                int64_t* __restrict__ col,
                                const int64_t k,
                                const int64_t n,
                                const int64_t m,
                                const int64_t dim,
                                const int64_t num_examples,
                                const bool cosine) {
  const int64_t n_y = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_y >= m)
    return;

  const int64_t example_idx = get_example_idx(n_y, ptr_y, num_examples);

  scalar_t best_dist[100];
  int64_t best_idx[100];

  for (int e = 0; e < k; e++) {
    best_dist[e] = (scalar_t)1e10;
    best_idx[e] = -1;
  }

  for (int64_t n_x = ptr_x[example_idx]; n_x < ptr_x[example_idx + 1]; n_x++) {
    scalar_t tmp_dist = 0;

    if (cosine) {
      tmp_dist = Cosine<scalar_t>::dot(x, y, n_x, n_y, dim) /
                 (Cosine<scalar_t>::norm(x, n_x, dim) *
                  Cosine<scalar_t>::norm(y, n_y, dim));
      tmp_dist = (scalar_t)1. - tmp_dist;
    } else {
      for (int64_t d = 0; d < dim; d++) {
        scalar_t diff = x[n_x * dim + d] - y[n_y * dim + d];
        tmp_dist += diff * diff;
      }
    }

    for (int64_t e1 = 0; e1 < k; e1++) {
      if (scalar_gt(best_dist[e1], tmp_dist)) {
        for (int64_t e2 = k - 1; e2 > e1; e2--) {
          best_dist[e2] = best_dist[e2 - 1];
          best_idx[e2] = best_idx[e2 - 1];
        }
        best_dist[e1] = tmp_dist;
        best_idx[e1] = n_x;
        break;
      }
    }
  }

  for (int64_t e = 0; e < k; e++) {
    row[n_y * k + e] = n_y;
    col[n_y * k + e] = best_idx[e];
  }
}

at::Tensor knn_cuda(const at::Tensor& x,
                    const at::Tensor& y,
                    const std::optional<at::Tensor>& ptr_x,
                    const std::optional<at::Tensor>& ptr_y,
                    int64_t k,
                    bool cosine,
                    int64_t num_workers) {
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), "Inputs must be CUDA tensors");
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous(),
              "Inputs must be contiguous");
  TORCH_CHECK(k <= 100, "`k` must be <= 100");

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

  auto row = at::empty({y.size(0) * k}, ptr_y_v.value().options());
  auto col = at::full({y.size(0) * k}, -1, ptr_y_v.value().options());

  dim3 BLOCKS((y.size(0) + KNN_THREADS - 1) / KNN_THREADS);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::Half, x.scalar_type(), "knn_cuda", [&] {
        knn_cuda_kernel<scalar_t><<<BLOCKS, KNN_THREADS, 0, stream>>>(
            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
            ptr_x_v.value().data_ptr<int64_t>(),
            ptr_y_v.value().data_ptr<int64_t>(), row.data_ptr<int64_t>(),
            col.data_ptr<int64_t>(), k, x.size(0), y.size(0), x.size(1),
            ptr_x_v.value().numel() - 1, cosine);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  auto mask = col != -1;
  return at::stack({row.masked_select(mask), col.masked_select(mask)}, 0);
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::knn"), TORCH_FN(knn_cuda));
}

}  // namespace ops
}  // namespace pyg
