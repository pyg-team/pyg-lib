#include "subgraph.h"
#include <pyg_lib/csrc/utils/hetero_dispatch.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

#include <functional>

namespace pyg {
namespace sampler {

template <typename T>
std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
subgraph_with_mapper(const at::Tensor& rowptr,
                     const at::Tensor& col,
                     const at::Tensor& nodes,
                     const Mapper<T>& mapper,
                     const bool return_edge_id) {
  const auto num_nodes = rowptr.size(0) - 1;
  const auto out_rowptr = rowptr.new_empty({nodes.size(0) + 1});
  at::Tensor out_col;
  c10::optional<at::Tensor> out_edge_id = c10::nullopt;

  AT_DISPATCH_INTEGRAL_TYPES(
      nodes.scalar_type(), "subgraph_kernel_with_mapper", [&] {
        const auto rowptr_data = rowptr.data_ptr<scalar_t>();
        const auto col_data = col.data_ptr<scalar_t>();
        const auto nodes_data = nodes.data_ptr<scalar_t>();

        // We first iterate over all nodes and collect information about the
        // number of edges in the induced subgraph.
        const auto deg = rowptr.new_empty({nodes.size(0)});
        auto deg_data = deg.data_ptr<scalar_t>();
        auto grain_size = at::internal::GRAIN_SIZE;
        at::parallel_for(
            0, nodes.size(0), grain_size, [&](int64_t _s, int64_t _e) {
              for (size_t i = _s; i < _e; ++i) {
                const auto v = nodes_data[i];
                // Iterate over all neighbors and check if they are part of
                // `nodes`:
                scalar_t d = 0;
                for (size_t j = rowptr_data[v]; j < rowptr_data[v + 1]; ++j) {
                  if (mapper.exists(col_data[j]))
                    d++;
                }
                deg_data[i] = d;
              }
            });

        auto out_rowptr_data = out_rowptr.data_ptr<scalar_t>();
        out_rowptr_data[0] = 0;
        auto tmp = out_rowptr.narrow(0, 1, nodes.size(0));
        at::cumsum_out(tmp, deg, /*dim=*/0);

        out_col = col.new_empty({out_rowptr_data[nodes.size(0)]});
        auto out_col_data = out_col.data_ptr<scalar_t>();
        scalar_t* out_edge_id_data;
        if (return_edge_id) {
          out_edge_id = col.new_empty({out_rowptr_data[nodes.size(0)]});
          out_edge_id_data = out_edge_id.value().data_ptr<scalar_t>();
        }

        // Customize `grain_size` based on the work each thread does (it will
        // need to find `col.size(0) / nodes.size(0)` neighbors on average).
        // TODO Benchmark this customization
        grain_size = std::max<int64_t>(out_col.size(0) / nodes.size(0), 1);
        grain_size = at::internal::GRAIN_SIZE / grain_size;
        at::parallel_for(
            0, nodes.size(0), grain_size, [&](int64_t _s, int64_t _e) {
              for (scalar_t i = _s; i < _e; ++i) {
                const auto v = nodes_data[i];
                // Iterate over all neighbors and check if they
                // are part of `nodes`:
                scalar_t offset = out_rowptr_data[i];
                for (scalar_t j = rowptr_data[v]; j < rowptr_data[v + 1]; ++j) {
                  const auto w = mapper.map(col_data[j]);
                  if (w >= 0) {
                    out_col_data[offset] = w;
                    if (return_edge_id)
                      out_edge_id_data[offset] = j;
                    offset++;
                  }
                }
              }
            });
      });

  return std::make_tuple(out_rowptr, out_col, out_edge_id);
}

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
  TORCH_CHECK(rowptr.is_cpu(), "'rowptr' must be a CPU tensor");
  TORCH_CHECK(col.is_cpu(), "'col' must be a CPU tensor");
  TORCH_CHECK(src_nodes.is_cpu(), "'src_nodes' must be a CPU tensor");
  TORCH_CHECK(dst_nodes.is_cpu(), "'dst_nodes' must be a CPU tensor");

  const auto num_nodes = rowptr.size(0) - 1;
  at::Tensor out_rowptr, out_col;
  c10::optional<at::Tensor> out_edge_id;

  AT_DISPATCH_INTEGRAL_TYPES(
      src_nodes.scalar_type(), "subgraph_bipartite", [&] {
        // TODO: at::max parallel but still a little expensive
        Mapper<scalar_t> mapper(at::max(col).item<scalar_t>() + 1,
                                dst_nodes.size(0));
        mapper.fill(dst_nodes);

        auto res = subgraph_with_mapper<scalar_t>(rowptr, col, src_nodes,
                                                  mapper, return_edge_id);
        out_rowptr = std::get<0>(res);
        out_col = std::get<1>(res);
        out_edge_id = std::get<2>(res);
      });

  return {out_rowptr, out_col, out_edge_id};
}

c10::Dict<utils::EdgeType,
          std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>>
hetero_subgraph(const utils::EdgeTensorDict& rowptr,
                const utils::EdgeTensorDict& col,
                const utils::NodeTensorDict& src_nodes,
                const utils::NodeTensorDict& dst_nodes,
                const c10::Dict<utils::EdgeType, bool>& return_edge_id) {
  // Define the bipartite implementation as a std function to pass the type
  // check
  std::function<std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>(
      const at::Tensor&, const at::Tensor&, const at::Tensor&,
      const at::Tensor&, bool)>
      func = subgraph_bipartite;

  // Construct an operator
  utils::HeteroDispatchOp<decltype(func)> op(rowptr, col, func);

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
  return op(src_nodes_arg, dst_nodes_arg, edge_id_arg);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::subgraph(Tensor rowptr, Tensor col, Tensor "
      "nodes, bool return_edge_id) -> (Tensor, Tensor, Tensor?)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::subgraph_bipartite(Tensor rowptr, Tensor col, Tensor "
      "src_nodes, Tensor dst_nodes, bool return_edge_id) -> (Tensor, Tensor, "
      "Tensor?)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::hetero_subgraph(Dict(str, Tensor) rowptr, Dict(str, "
      "Tensor) col, Dict(str, Tensor) nodes, Dict(str, bool) "
      "return_edge_id) -> Dict(str, (Tensor, Tensor, Tensor?))"));
}

}  // namespace sampler
}  // namespace pyg
