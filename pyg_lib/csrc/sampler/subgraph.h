#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace sampler {

// Returns the induced subgraph of the graph given by `(rowptr, col)`,
// containing only the nodes in `nodes`.
// Returns: (rowptr, col, edge_id)
PYG_API std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> subgraph(
    const at::Tensor& rowptr,
    const at::Tensor& col,
    const at::Tensor& nodes,
    bool return_edge_id = true);

}  // namespace sampler
}  // namespace pyg
