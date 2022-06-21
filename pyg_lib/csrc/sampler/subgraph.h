#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include "pyg_lib/csrc/macros.h"
#include "pyg_lib/csrc/sampler/cpu/mapper.h"
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

// A bipartite version of the above function.
PYG_API std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
subgraph_bipartite(const at::Tensor& rowptr,
                   const at::Tensor& col,
                   const at::Tensor& src_nodes,
                   const at::Tensor& dst_nodes,
                   const bool return_edge_id);

// A heterogeneous version of the above function.
// Returns a dict from each relation type to its result
PYG_API c10::Dict<utils::EdgeType,
                  std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>>
hetero_subgraph(const utils::EdgeTensorDict& rowptr,
                const utils::EdgeTensorDict& col,
                const utils::NodeTensorDict& src_nodes,
                const utils::NodeTensorDict& dst_nodes,
                const c10::Dict<utils::EdgeType, bool>& return_edge_id);

}  // namespace sampler
}  // namespace pyg
