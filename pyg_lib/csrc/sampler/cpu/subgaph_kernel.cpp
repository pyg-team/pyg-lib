#include <ATen/ATen.h>
#include <torch/library.h>

namespace pyg {
namespace sampler {

namespace {

std::tuple<at::Tensor, at::Tensor, at::Tensor> subgraph_kernel(
    const at::Tensor& rowptr,
    const at::Tensor& col,
    const at::Tensor& nodes) {
  TORCH_CHECK(rowptr.is_cpu(), "'rowptr' must be a CPU tensor");
  TORCH_CHECK(col.is_cpu(), "'col' must be a CPU tensor");
  TORCH_CHECK(nodes.is_cpu(), "'nodes' must be a CPU tensor");

  return std::make_tuple(rowptr, col, nodes);
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::subgraph"), TORCH_FN(subgraph_kernel));
}

}  // namespace sampler
}  // namespace pyg
