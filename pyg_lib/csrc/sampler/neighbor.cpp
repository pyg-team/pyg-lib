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
                const c10::optional<at::Tensor>& time,
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
  return op.call(rowptr, col, seed, num_neighbors, time, replace, directed,
                 disjoint, return_edge_id);
}

std::tuple<c10::Dict<rel_type, at::Tensor>,
           c10::Dict<rel_type, at::Tensor>,
           c10::Dict<node_type, at::Tensor>,
           c10::optional<c10::Dict<rel_type, at::Tensor>>>
hetero_neighbor_sample(
    const std::vector<node_type>& node_types,
    const std::vector<edge_type>& edge_types,
    const c10::Dict<rel_type, at::Tensor>& rowptr_dict,
    const c10::Dict<rel_type, at::Tensor>& col_dict,
    const c10::Dict<node_type, at::Tensor>& seed_dict,
    const c10::Dict<rel_type, std::vector<int64_t>>& num_neighbors_dict,
    const c10::optional<c10::Dict<node_type, at::Tensor>>& time_dict,
    bool replace,
    bool directed,
    bool disjoint,
    bool return_edge_id) {
  // TODO (matthias) Add TensorArg definitions and type checks.
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::hetero_neighbor_sample_cpu", "")
                       .typed<decltype(hetero_neighbor_sample)>();
  return op.call(node_types, edge_types, rowptr_dict, col_dict, seed_dict,
                 num_neighbors_dict, time_dict, replace, directed, disjoint,
                 return_edge_id);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::neighbor_sample(Tensor rowptr, Tensor col, Tensor seed, int[] "
      "num_neighbors, Tensor? time, bool replace, bool directed, bool "
      "disjoint, bool return_edge_id) -> (Tensor, Tensor, Tensor, Tensor?)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::hetero_neighbor_sample(str[] node_types, (str, str, str)[] "
      "edge_types, Dict(str, Tensor) rowptr_dict, Dict(str, Tensor) col_dict, "
      "Dict(str, Tensor) seed_dict, Dict(str, int[]) num_neighbors_dict, "
      "Dict(str, Tensor)? time_dict, bool replace, bool directed, bool "
      "disjoint, bool return_edge_id) -> (Dict(str, Tensor), Dict(str, "
      "Tensor), Dict(str, Tensor), Dict(str, Tensor)?)"));
}

}  // namespace sampler
}  // namespace pyg
