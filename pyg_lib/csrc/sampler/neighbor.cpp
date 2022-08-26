#include "neighbor.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace sampler {

std::tuple<at::Tensor, at::Tensor, at::Tensor, c10::optional<at::Tensor>>
neighbor_sample(const at::Tensor& rowptr,
                const at::Tensor& col,
                const at::Tensor& seed,
                const std::vector<int64_t>& num_neighbors,
                bool replace,
                bool directed,
                bool disjoint,
                bool return_edge_id) {
  at::TensorArg rowptr_t{rowptr, "rowtpr", 1};
  at::TensorArg col_t{col, "col", 1};
  at::TensorArg seed_t{seed, "seed", 1};

  at::CheckedFrom c = "neighbor_sample";
  at::checkAllDefined(c, {rowptr_t, col_t, seed_t});
  at::checkAllSameType(c, {rowptr_t, col_t, seed_t});

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::neighbor_sample", "")
                       .typed<decltype(neighbor_sample)>();
  return op.call(rowptr, col, seed, num_neighbors, replace, directed, disjoint,
                 return_edge_id);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::neighbor_sample(Tensor rowptr, Tensor col, Tensor seed, int[] "
      "num_neighbors, bool replace, bool directed, bool disjoint, bool "
      "return_edge_id) -> (Tensor, Tensor, Tensor, Tensor?)"));
}

}  // namespace sampler
}  // namespace pyg
