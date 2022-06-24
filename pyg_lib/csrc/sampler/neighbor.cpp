#include "neighbor.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace sampler {

std::tuple<at::Tensor, at::Tensor, at::Tensor, c10::optional<at::Tensor>>
neighbor_sample(const at::Tensor& rowptr,
                const at::Tensor& col,
                const at::Tensor& seed,
                const std::vector<int64_t> num_neighbors,
                bool replace,
                bool directed,
                bool isolated,
                bool return_edge_id) {
  // TODO (matthias) Add TensorArg definitions.
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::neighbor_sample", "")
                       .typed<decltype(neighbor_sample)>();
  return op.call(rowptr, col, seed, num_neighbors, replace, directed, isolated,
                 return_edge_id);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::neighbor_sample(Tensor rowptr, Tensor col, Tensor seed, int[] "
      "num_neighbors, bool replace, bool directed, bool isolated, bool "
      "return_edge_id) -> (Tensor, Tensor, Tensor, Tensor?)"));
}

}  // namespace sampler
}  // namespace pyg
