#include "subgraph.h"
#include <pyg_lib/csrc/utils/hetero_dispatch.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

#include <functional>

namespace pyg {
namespace sampler {

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> subgraph(
    const at::Tensor& rowptr,
    const at::Tensor& col,
    const at::Tensor& nodes,
    const bool return_edge_id) {
  at::TensorArg rowptr_t{rowptr, "rowptr", 1};
  at::TensorArg col_t{col, "col", 1};
  at::TensorArg nodes_t{nodes, "nodes", 1};

  at::CheckedFrom c = "subgraph";
  at::checkAllDefined(c, {rowptr_t, col_t, nodes_t});
  at::checkAllSameType(c, {rowptr_t, col_t, nodes_t});

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::subgraph", "")
                       .typed<decltype(subgraph)>();
  return op.call(rowptr, col, nodes, return_edge_id);
}

c10::Dict<utils::edge_t,
          std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>>
hetero_subgraph(const utils::edge_tensor_dict_t& rowptr,
                const utils::edge_tensor_dict_t& col,
                const utils::node_tensor_dict_t& nodes,
                const c10::Dict<utils::edge_t, bool>& return_edge_id) {
  // Define the homogeneous implementation as a std function to pass the type
  // check
  std::function<std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>(
      const at::Tensor&, const at::Tensor&, const at::Tensor&, bool)>
      func = subgraph;

  // Construct an operator
  utils::HeteroDispatchOp<decltype(func)> op(rowptr, col, func);

  // Construct dispatchable arguments
  // TODO: We filter source node by assuming hetero graph is a dict of homo
  // graph here; both source and destination nodes should be considered when
  // filtering a bipartite graph
  utils::HeteroDispatchArg<utils::node_tensor_dict_t, at::Tensor,
                           utils::NodeSrcMode>
      nodes_arg(nodes);
  utils::HeteroDispatchArg<c10::Dict<utils::edge_t, bool>, bool,
                           utils::EdgeMode>
      edge_id_arg(return_edge_id);
  return op(nodes_arg, edge_id_arg);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::subgraph(Tensor rowptr, Tensor col, Tensor "
      "nodes, bool return_edge_id) -> (Tensor, Tensor, Tensor?)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::hetero_subgraph(Dict(str, Tensor) rowptr, Dict(str, "
      "Tensor) col, Dict(str, Tensor) nodes, Dict(str, bool) "
      "return_edge_id) -> Dict(str, (Tensor, Tensor, Tensor?))"));
}

}  // namespace sampler
}  // namespace pyg
