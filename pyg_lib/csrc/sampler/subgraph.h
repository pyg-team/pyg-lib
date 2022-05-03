#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"
#include "pyg_lib/csrc/utils/types.h"

namespace pyg {
namespace sampler {

// Returns the induced subgraph of the graph given by `(rowptr, col)`,
// containing only the nodes in `nodes`.
// Returns: (rowptr, col, edge_id)
PYG_API std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> subgraph(
    const at::Tensor& rowptr,
    const at::Tensor& col,
    const at::Tensor& nodes,
    const bool return_edge_id = true);

// A heterogeneous version of the above function.
// Returns a dict from each relation type to its result
PYG_API c10::Dict<utils::RELATION_TYPE,
                  std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>>
hetero_subgraph(const utils::HETERO_TENSOR_TYPE& rowptr,
                const utils::HETERO_TENSOR_TYPE& col,
                const utils::HETERO_TENSOR_TYPE& nodes,
                const c10::Dict<utils::RELATION_TYPE, bool> return_edge_id);

}  // namespace sampler
}  // namespace pyg
