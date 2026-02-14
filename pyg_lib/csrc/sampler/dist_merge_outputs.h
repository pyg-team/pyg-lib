#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"
#include "pyg_lib/csrc/utils/types.h"

namespace pyg {
namespace sampler {

// For distributed training purposes. Merges sampler outputs from different
// partitions, so that they are sorted according to the sampling order.
// Removes seed nodes from sampled nodes and calculates how many neighbors
// were sampled by each source node based on the cumulative sum of sampled
// neighbors for each input node.
// Returns the unified node, edge and batch indices as well as the merged
// cumulative sum of sampled neighbors.
PYG_API
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
    const c10::optional<at::Tensor>& batch = c10::nullopt,
    bool disjoint = false);

}  // namespace sampler
}  // namespace pyg
