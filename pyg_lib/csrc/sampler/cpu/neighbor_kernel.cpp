#include <ATen/ATen.h>
#include <torch/library.h>

namespace pyg {
namespace sampler {

namespace {

std::tuple<at::Tensor, at::Tensor, at::Tensor, c10::optional<at::Tensor>>
neighbor_sample_kernel(const at::Tensor& rowptr,
                       const at::Tensor& col,
                       const at::Tensor& seed,
                       const std::vector<int64_t> num_neighbors,
                       bool replace,
                       bool directed,
                       bool isolated,
                       bool return_edge_id) {
  return std::make_tuple(rowptr, col, seed, at::nullopt);
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::neighbor_sample"),
         TORCH_FN(neighbor_sample_kernel));
}

}  // namespace sampler
}  // namespace pyg
