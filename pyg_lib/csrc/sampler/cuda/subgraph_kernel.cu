#include <ATen/ATen.h>
#include <torch/library.h>

#include "pyg_lib/csrc/utils/cuda/helpers.h"

namespace pyg {
namespace sampler {

namespace {

template <typename scalar_t>
__global__ void subgraph_deg_kernel_impl(
    const scalar_t* __restrict__ rowptr_data,
    const scalar_t* __restrict__ col_data,
    const scalar_t* __restrict__ nodes_data,
    const scalar_t* __restrict__ to_local_node_data,
    scalar_t* __restrict__ out_data,
    int64_t num_nodes) {
  CUDA_1D_KERNEL_LOOP(scalar_t, thread_idx, WARP * num_nodes) {
    scalar_t i = thread_idx / WARP;
    scalar_t lane = thread_idx % WARP;

    auto v = nodes_data[i];

    scalar_t deg = 0;
    for (size_t j = rowptr_data[v] + lane; j < rowptr_data[v + 1]; j += WARP) {
      if (to_local_node_data[col_data[j]] >= 0)  // contiguous access
        deg++;
    }

    for (size_t offset = 16; offset > 0; offset /= 2)  // warp-level reduction
      deg += __shfl_down_sync(FULL_WARP_MASK, deg, offset);

    if (lane == 0)
      out_data[i] = deg;
  }
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

  // We maintain a O(num_nodes) vector to map global node indices to local ones.
  // TODO Can we do this without O(num_nodes) storage requirement?
  // TODO Consider caching this tensor  across consecutive runs?
  const auto to_local_node = nodes.new_full({rowptr.size(0) - 1}, -1);
  const auto arange = at::arange(nodes.size(0), nodes.options());
  to_local_node.index_copy_(/*dim=*/0, nodes, arange);

  const auto deg = nodes.new_empty({nodes.size(0)});
  const auto out_rowptr = rowptr.new_zeros({nodes.size(0) + 1});
  at::Tensor out_col;
  c10::optional<at::Tensor> out_edge_id = c10::nullopt;

  AT_DISPATCH_INTEGRAL_TYPES(nodes.scalar_type(), "subgraph_kernel", [&] {
    const auto rowptr_data = rowptr.data_ptr<scalar_t>();
    const auto col_data = col.data_ptr<scalar_t>();
    const auto nodes_data = nodes.data_ptr<scalar_t>();
    const auto to_local_node_data = to_local_node.data_ptr<scalar_t>();
    auto deg_data = deg.data_ptr<scalar_t>();

    // Compute induced subgraph degree, parallelize with 32 threads per node:
    subgraph_deg_kernel_impl<<<pyg::utils::blocks(WARP * nodes.size(0)),
                               pyg::utils::threads(), 0, stream>>>(
        rowptr_data, col_data, nodes_data, to_local_node_data, deg_data,
        nodes.size(0));

    auto tmp = out_rowptr.narrow(0, 1, nodes.size(0));
    at::cumsum_out(tmp, deg, /*dim=*/0);
  });

  return std::make_tuple(out_rowptr, deg, deg);
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::subgraph"), TORCH_FN(subgraph_kernel));
}

}  // namespace sampler
}  // namespace pyg
