#include "dist_merge_outputs.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

#include "pyg_lib/csrc/utils/check.h"

namespace pyg {
namespace sampler {

std::tuple<at::Tensor,
           c10::optional<at::Tensor>,
           c10::optional<at::Tensor>,
           std::vector<int64_t>>
merge_sampler_outputs(
    const std::vector<at::Tensor>& nodes,
    const std::vector<std::vector<int64_t>>& cumsum_neighbors_per_node,
    const std::vector<int64_t>& partition_ids,
    const std::vector<int64_t>& partition_orders,
    const int64_t partitions_num,
    const int64_t num_neighbors,
    const c10::optional<std::vector<at::Tensor>>& edge_ids,
    const c10::optional<at::Tensor>& batch,
    bool disjoint,
    bool with_edge) {
  std::vector<at::TensorArg> nodes_args;
  pyg::utils::fill_tensor_args(nodes_args, nodes, "nodes", 0);

  at::CheckedFrom c{"merge_sampler_outputs"};
  at::checkAllDefined(c, {nodes_args});

  TORCH_CHECK(partition_ids.size() == partition_orders.size(),
              "Each id must be assigned a sampling order'");

  if (disjoint) {
    TORCH_CHECK(batch.has_value(),
                "I case of disjoint sampling batch needs to be specified");
    TORCH_CHECK(batch.value().numel() == partition_ids.size(),
                "Each src node must belong to a subgraph'");
  }

  if (with_edge)
    TORCH_CHECK(edge_ids.has_value(), "No edge ids specified");

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::merge_sampler_outputs", "")
                       .typed<decltype(merge_sampler_outputs)>();
  return op.call(nodes, cumsum_neighbors_per_node, partition_ids,
                 partition_orders, partitions_num, num_neighbors, edge_ids,
                 batch, disjoint, with_edge);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::merge_sampler_outputs(Tensor[] nodes, "
      "int[][] cumsum_neighbors_per_node, int[] partition_ids, int[] "
      "partition_orders, int partitions_num, int num_neighbors, Tensor[]? "
      "edge_ids, Tensor? batch, bool disjoint, bool with_edge) -> (Tensor, "
      "Tensor?, Tensor?, int[])"));
}

}  // namespace sampler
}  // namespace pyg
