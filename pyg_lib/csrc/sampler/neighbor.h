#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"

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
                bool replace = false,
                bool directed = true,
                bool disjoint = false,
                bool return_edge_id = true);

}  // namespace sampler
}  // namespace pyg
