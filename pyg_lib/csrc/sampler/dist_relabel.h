#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"
#include "pyg_lib/csrc/utils/types.h"

namespace pyg {
namespace sampler {

// Relabels global indices from `sampled_nodes_with_duplicates` to the local
// subtree/subgraph indices in the homogeneous graph.
// Seed nodes should not be included.
// Returns (row, col).
PYG_API
std::tuple<at::Tensor, at::Tensor> relabel_neighborhood(
    const at::Tensor& seed,
    const at::Tensor& sampled_nodes_with_duplicates,
    const std::vector<int64_t>& num_sampled_neighbors_per_node,
    const int64_t num_nodes,
    const c10::optional<at::Tensor>& batch = c10::nullopt,
    bool csc = false,
    bool disjoint = false);

// Relabels global indices from `sampled_nodes_with_duplicates` to the local
// subtree/subgraph indices in the heterogeneous graph.
// Seed nodes should not be included.
// Returns (row_dict, col_dict).
PYG_API
std::tuple<c10::Dict<rel_type, at::Tensor>, c10::Dict<rel_type, at::Tensor>>
hetero_relabel_neighborhood(
    const std::vector<node_type>& node_types,
    const std::vector<edge_type>& edge_types,
    const c10::Dict<node_type, at::Tensor>& seed_dict,
    const c10::Dict<node_type, at::Tensor>& sampled_nodes_with_duplicates_dict,
    const c10::Dict<rel_type, std::vector<std::vector<int64_t>>>&
        num_sampled_neighbors_per_node_dict,
    const c10::Dict<node_type, int64_t>& num_nodes_dict,
    const c10::optional<c10::Dict<node_type, at::Tensor>>& batch_dict =
        c10::nullopt,
    bool csc = false,
    bool disjoint = false);

}  // namespace sampler
}  // namespace pyg
