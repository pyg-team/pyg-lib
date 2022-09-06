#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"
#include "pyg_lib/csrc/utils/types.h"

namespace pyg {
namespace sampler {

// Recursively samples neighbors from all node indices in `seed`
// in the graph given by `(rowptr, col)`.
// Returns: (row, col, node_id, edge_id)
PYG_API
std::tuple<at::Tensor, at::Tensor, at::Tensor, c10::optional<at::Tensor>>
neighbor_sample(const at::Tensor& rowptr,
                const at::Tensor& col,
                const at::Tensor& seed,
                const std::vector<int64_t>& num_neighbors,
                const c10::optional<at::Tensor>& time = c10::nullopt,
                bool replace = false,
                bool directed = true,
                bool disjoint = false,
                bool return_edge_id = true);

// Recursively samples neighbors from all node indices in `seed_dict`
// in the heterogeneous graph given by `(rowptr_dict, col_dict)`.
// Returns: (row_dict, col_dict, node_id_dict, edge_id_dict)
PYG_API
std::tuple<c10::Dict<rel_t, at::Tensor>,
           c10::Dict<rel_t, at::Tensor>,
           c10::Dict<node_t, at::Tensor>,
           c10::optional<c10::Dict<rel_t, at::Tensor>>>
hetero_neighbor_sample(
    const std::vector<node_t>& node_types,
    const std::vector<edge_t>& edge_types,
    const c10::Dict<rel_t, at::Tensor>& rowptr_dict,
    const c10::Dict<rel_t, at::Tensor>& col_dict,
    const c10::Dict<node_t, at::Tensor>& seed_dict,
    const c10::Dict<rel_t, std::vector<int64_t>>& num_neighbors_dict,
    const c10::optional<c10::Dict<node_t, at::Tensor>>& time_dict =
        c10::nullopt,
    bool replace = false,
    bool directed = true,
    bool disjoint = false,
    bool return_edge_id = true);

}  // namespace sampler
}  // namespace pyg
