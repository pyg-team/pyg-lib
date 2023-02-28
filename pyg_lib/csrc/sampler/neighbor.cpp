#include "neighbor.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

#include "pyg_lib/csrc/utils/check.h"

namespace pyg {
namespace sampler {

std::tuple<at::Tensor,
           at::Tensor,
           at::Tensor,
           c10::optional<at::Tensor>,
           std::vector<int64_t>,
           std::vector<int64_t>>
neighbor_sample(const at::Tensor& rowptr,
                const at::Tensor& col,
                const at::Tensor& seed,
                const std::vector<int64_t>& num_neighbors,
                const c10::optional<at::Tensor>& time,
                const c10::optional<at::Tensor>& seed_time,
                bool csc,
                bool replace,
                bool directed,
                bool disjoint,
                std::string temporal_strategy,
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
  return op.call(rowptr, col, seed, num_neighbors, time, seed_time, csc,
                 replace, directed, disjoint, temporal_strategy,
                 return_edge_id);
}

std::tuple<c10::Dict<rel_type, at::Tensor>,
           c10::Dict<rel_type, at::Tensor>,
           c10::Dict<node_type, at::Tensor>,
           c10::optional<c10::Dict<rel_type, at::Tensor>>,
           c10::Dict<node_type, std::vector<int64_t>>,
           c10::Dict<rel_type, std::vector<int64_t>>>
hetero_neighbor_sample(
    const std::vector<node_type>& node_types,
    const std::vector<edge_type>& edge_types,
    const c10::Dict<rel_type, at::Tensor>& rowptr_dict,
    const c10::Dict<rel_type, at::Tensor>& col_dict,
    const c10::Dict<node_type, at::Tensor>& seed_dict,
    const c10::Dict<rel_type, std::vector<int64_t>>& num_neighbors_dict,
    const c10::optional<c10::Dict<node_type, at::Tensor>>& time_dict,
    const c10::optional<c10::Dict<node_type, at::Tensor>>& seed_time_dict,
    bool csc,
    bool replace,
    bool directed,
    bool disjoint,
    std::string temporal_strategy,
    bool return_edge_id) {
  TORCH_CHECK(rowptr_dict.size() == col_dict.size(),
              "Number of edge types in 'rowptr_dict' and 'col_dict' must match")

  std::vector<at::TensorArg> rowptr_dict_args;
  std::vector<at::TensorArg> col_dict_args;
  std::vector<at::TensorArg> seed_dict_args;
  pyg::utils::fill_tensor_args(rowptr_dict_args, rowptr_dict, "rowptr_dict", 0);
  pyg::utils::fill_tensor_args(col_dict_args, col_dict, "col_dict", 0);
  pyg::utils::fill_tensor_args(seed_dict_args, seed_dict, "seed_dict", 0);
  at::CheckedFrom c{"hetero_neighbor_sample"};

  at::checkAllDefined(c, rowptr_dict_args);
  at::checkAllDefined(c, col_dict_args);
  at::checkAllDefined(c, seed_dict_args);
  at::checkAllSameType(c, rowptr_dict_args);
  at::checkAllSameType(c, col_dict_args);
  at::checkAllSameType(c, seed_dict_args);
  at::checkSameType(c, rowptr_dict_args[0], col_dict_args[0]);
  at::checkSameType(c, rowptr_dict_args[0], seed_dict_args[0]);

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::hetero_neighbor_sample", "")
                       .typed<decltype(hetero_neighbor_sample)>();
  return op.call(node_types, edge_types, rowptr_dict, col_dict, seed_dict,
                 num_neighbors_dict, time_dict, seed_time_dict, csc, replace,
                 directed, disjoint, temporal_strategy, return_edge_id);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::neighbor_sample(Tensor rowptr, Tensor col, Tensor seed, int[] "
      "num_neighbors, Tensor? time = None, Tensor? seed_time = None, bool csc "
      "= False, bool replace = False, bool directed = True, bool disjoint = "
      "False, str temporal_strategy = 'uniform', bool return_edge_id = True) "
      "-> (Tensor, Tensor, Tensor, Tensor?, int[], int[])"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::hetero_neighbor_sample(str[] node_types, (str, str, str)[] "
      "edge_types, Dict(str, Tensor) rowptr_dict, Dict(str, Tensor) col_dict, "
      "Dict(str, Tensor) seed_dict, Dict(str, int[]) num_neighbors_dict, "
      "Dict(str, Tensor)? time_dict = None, Dict(str, Tensor)? seed_time_dict "
      "= None, bool csc = False, bool replace = False, bool directed = True, "
      "bool disjoint = False, str temporal_strategy = 'uniform', bool "
      "return_edge_id = True) -> (Dict(str, Tensor), Dict(str, Tensor), "
      "Dict(str, Tensor), Dict(str, Tensor)?, Dict(str, int[]), "
      "Dict(str, int[]))"));
}

}  // namespace sampler
}  // namespace pyg
