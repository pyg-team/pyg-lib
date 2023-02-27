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
                const c10::optional<at::Tensor>& time = c10::nullopt,
                const c10::optional<at::Tensor>& seed_time = c10::nullopt,
                bool csc = false,
                bool replace = false,
                bool directed = true,
                bool disjoint = false,
                std::string strategy = "uniform",
                bool return_edge_id = true);

// Recursively samples neighbors from all node indices in `seed_dict`
// in the heterogeneous graph given by `(rowptr_dict, col_dict)`.
// Returns: (row_dict, col_dict, node_id_dict, edge_id_dict)
PYG_API
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
    const c10::optional<c10::Dict<node_type, at::Tensor>>& time_dict =
        c10::nullopt,
    const c10::optional<c10::Dict<node_type, at::Tensor>>& seed_time_dict =
        c10::nullopt,
    bool csc = false,
    bool replace = false,
    bool directed = true,
    bool disjoint = false,
    std::string strategy = "uniform",
    bool return_edge_id = true);

}  // namespace sampler
}  // namespace pyg
