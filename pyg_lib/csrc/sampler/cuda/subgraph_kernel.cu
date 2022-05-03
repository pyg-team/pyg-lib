#include <ATen/ATen.h>
#include <torch/library.h>

#include "pyg_lib/csrc/utils/cuda/helpers.h"

namespace pyg {
namespace sampler {

namespace {

template <typename scalar_t, bool return_edge_id>
__global__ void subgraph_walk_kernel_impl(
    const scalar_t* __restrict__ rowptr_data,
    const scalar_t* __restrict__ col_data,
    const scalar_t* __restrict__ nodes_data,
    const float* __restrict__ rand_data,
    scalar_t* __restrict__ out_data,
    int64_t num_seeds,
    int64_t walk_length) {
  CUDA_1D_KERNEL_LOOP(scalar_t, i, num_seeds) {}
}

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> subgraph_kernel(
    const at::Tensor& rowptr,
    const at::Tensor& col,
    const at::Tensor& nodes,
    const bool return_edge_id) {
  TORCH_CHECK(rowptr.is_cuda(), "'rowptr' must be a CUDA tensor");
  TORCH_CHECK(col.is_cuda(), "'col' must be a CUDA tensor");
  TORCH_CHECK(nodes.is_cuda(), "'nodes' must be a CUDA tensor");

  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_INTEGRAL_TYPES(nodes.scalar_type(), "subgraph_kernel", [&] {});

  return std::make_tuple(rowptr, rowptr, rowptr);
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::subgraph"), TORCH_FN(subgraph_kernel));
}

}  // namespace sampler
}  // namespace pyg
