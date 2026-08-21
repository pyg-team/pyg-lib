#include "../cluster.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

namespace {

#define THREADS 1024
#define BLOCKS(N) ((N) + THREADS - 1) / THREADS

template <typename scalar_t>
__global__ void grid_cluster_cuda_kernel(const scalar_t* pos,
                                         const scalar_t* size,
                                         const scalar_t* start,
                                         const scalar_t* end,
                                         int64_t* out,
                                         int64_t D,
                                         int64_t numel) {
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_idx < numel) {
    int64_t c = 0, k = 1;
    for (int64_t d = 0; d < D; d++) {
      scalar_t p = pos[thread_idx * D + d] - start[d];
      c += (int64_t)(p / size[d]) * k;
      k *= (int64_t)((end[d] - start[d]) / size[d]) + 1;
    }
    out[thread_idx] = c;
  }
}

at::Tensor grid_cluster_cuda(const at::Tensor& pos,
                             const at::Tensor& size,
                             const std::optional<at::Tensor>& optional_start,
                             const std::optional<at::Tensor>& optional_end) {
  TORCH_CHECK(pos.is_cuda(), "pos must be a CUDA tensor");
  TORCH_CHECK(pos.is_contiguous(), "pos must be contiguous");

  auto N = pos.size(0);
  auto D = pos.size(1);

  at::Tensor start;
  if (optional_start.has_value())
    start = optional_start.value().contiguous();
  else
    start = std::get<0>(pos.min(0));

  at::Tensor end;
  if (optional_end.has_value())
    end = optional_end.value().contiguous();
  else
    end = std::get<0>(pos.max(0));

  auto out = at::empty({N}, pos.options().dtype(at::kLong));

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, pos.scalar_type(),
      "grid_cluster_cuda", [&] {
        grid_cluster_cuda_kernel<scalar_t><<<BLOCKS(N), THREADS, 0, stream>>>(
            pos.data_ptr<scalar_t>(), size.data_ptr<scalar_t>(),
            start.data_ptr<scalar_t>(), end.data_ptr<scalar_t>(),
            out.data_ptr<int64_t>(), D, N);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::grid_cluster"),
         TORCH_FN(grid_cluster_cuda));
}

}  // namespace ops
}  // namespace pyg
