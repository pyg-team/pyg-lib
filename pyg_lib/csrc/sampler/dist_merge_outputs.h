#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"
#include "pyg_lib/csrc/utils/types.h"

namespace pyg {
namespace sampler {

// For distributed training purpose. Merges samplers outputs from different
// partitions, so that they are sorted according to the sampling order.
// Removes seed nodes from sampled nodes and calculates how many neighbors
// were sampled by each src node based on the :obj:`cumsum_neighbors_per_node`.
PYG_API
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
    const int64_t one_hop_num,
    const c10::optional<std::vector<at::Tensor>>& edge_ids = c10::nullopt,
    const c10::optional<at::Tensor>& batch = c10::nullopt,
    bool disjoint = false,
    bool with_edge = true);

}  // namespace sampler
}  // namespace pyg
