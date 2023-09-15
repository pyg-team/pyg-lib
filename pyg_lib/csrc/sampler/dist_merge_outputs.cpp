#include "dist_merge_outputs.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

#include "pyg_lib/csrc/utils/check.h"

namespace pyg {
namespace sampler {

std::tuple<at::Tensor,
           at::Tensor,
           c10::optional<at::Tensor>,
           std::vector<int64_t>>
merge_sampler_outputs(
    const std::vector<at::Tensor>& node_ids,
    const std::vector<at::Tensor>& edge_ids,
    const std::vector<std::vector<int64_t>>& cumsum_neighbors_per_node,
    const std::vector<int64_t>& partition_ids,
    const std::vector<int64_t>& partition_orders,
    const int64_t num_partitions,
    const int64_t num_neighbors,
    const c10::optional<at::Tensor>& batch,
    bool disjoint) {
  std::vector<at::TensorArg> node_ids_args;
  std::vector<at::TensorArg> edge_ids_args;
  pyg::utils::fill_tensor_args(node_ids_args, node_ids, "node_ids", 0);
  pyg::utils::fill_tensor_args(edge_ids_args, edge_ids, "edge_ids", 0);

  at::CheckedFrom c{"merge_sampler_outputs"};
  at::checkAllDefined(c, {node_ids_args});
  at::checkAllDefined(c, {edge_ids_args});

  TORCH_CHECK(partition_ids.size() == partition_orders.size(),
              "Every partition ID must be assigned a sampling order");

  if (disjoint) {
    TORCH_CHECK(batch.has_value(),
                "Disjoint sampling requires 'batch' to be specified");
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::merge_sampler_outputs", "")
                       .typed<decltype(merge_sampler_outputs)>();
  return op.call(node_ids, edge_ids, cumsum_neighbors_per_node, partition_ids,
                 partition_orders, num_partitions, num_neighbors, batch,
                 disjoint);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::merge_sampler_outputs(Tensor[] node_ids, Tensor[] edge_ids, "
      "int[][] cumsum_neighbors_per_node, int[] partition_ids, int[] "
      "partition_orders, int num_partitions, int num_neighbors, Tensor? "
      "batch, bool disjoint) -> (Tensor, Tensor, Tensor?, int[])"));
}

}  // namespace sampler
}  // namespace pyg
