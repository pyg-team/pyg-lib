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

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
subgraph_bipartite(const at::Tensor& rowptr,
                   const at::Tensor& col,
                   const at::Tensor& src_nodes,
                   const at::Tensor& dst_nodes,
                   const bool return_edge_id) {
  at::TensorArg rowptr_t{rowptr, "rowptr", 1};
  at::TensorArg col_t{col, "col", 1};
  at::TensorArg src_nodes_t{src_nodes, "src_nodes", 1};
  at::TensorArg dst_nodes_t{dst_nodes, "dst_nodes", 1};

  at::CheckedFrom c = "subgraph_bipartite";
  at::checkAllDefined(c, {rowptr_t, col_t, src_nodes_t, dst_nodes_t});
  at::checkAllSameType(c, {rowptr_t, col_t, src_nodes_t, dst_nodes_t});

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::subgraph_bipartite", "")
                       .typed<decltype(subgraph_bipartite)>();
  return op.call(rowptr, col, src_nodes, dst_nodes, return_edge_id);
}

c10::Dict<utils::EdgeType,
          std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>>
hetero_subgraph(const utils::EdgeTensorDict& rowptr,
                const utils::EdgeTensorDict& col,
                const utils::NodeTensorDict& src_nodes,
                const utils::NodeTensorDict& dst_nodes,
                const c10::Dict<utils::EdgeType, bool>& return_edge_id) {
  c10::Dict<utils::EdgeType,
            std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>>
      res;

  // Construct dispatchable arguments
  utils::HeteroDispatchArg<utils::NodeTensorDict, at::Tensor,
                           utils::NodeSrcMode>
      src_nodes_arg(src_nodes);
  utils::HeteroDispatchArg<utils::NodeTensorDict, at::Tensor,
                           utils::NodeDstMode>
      dst_nodes_arg(dst_nodes);
  utils::HeteroDispatchArg<c10::Dict<utils::EdgeType, bool>, bool,
                           utils::EdgeMode>
      edge_id_arg(return_edge_id);

  for (const auto& kv : rowptr) {
    const auto& edge_type = kv.key();
    bool pass = src_nodes_arg.filter_by_edge(edge_type) &&
                dst_nodes_arg.filter_by_edge(edge_type) &&
                edge_id_arg.filter_by_edge(edge_type);
    if (pass) {
      const auto& r = rowptr.at(edge_type);
      const auto& c = col.at(edge_type);
      res.insert(edge_type, subgraph_bipartite(
                                r, c, src_nodes_arg.value_by_edge(edge_type),
                                dst_nodes_arg.value_by_edge(edge_type),
                                edge_id_arg.value_by_edge(edge_type)));
    }
  }

  return res;
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::subgraph(Tensor rowptr, Tensor col, Tensor "
      "nodes, bool return_edge_id) -> (Tensor, Tensor, Tensor?)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::subgraph_bipartite(Tensor rowptr, Tensor col, Tensor "
      "src_nodes, Tensor dst_nodes, bool return_edge_id) -> (Tensor, Tensor, "
      "Tensor?)"));
  m.def("hetero_subgraph", hetero_subgraph);
}

}  // namespace sampler
}  // namespace pyg
