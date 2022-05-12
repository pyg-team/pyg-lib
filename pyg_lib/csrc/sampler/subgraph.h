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
PYG_API c10::Dict<utils::edge_t,
                  std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>>
hetero_subgraph(const utils::edge_tensor_dict_t& rowptr,
                const utils::edge_tensor_dict_t& col,
                const utils::node_tensor_dict_t& src_nodes,
                const utils::node_tensor_dict_t& dst_nodes,
                const c10::Dict<utils::edge_t, bool>& return_edge_id);

template <typename T>
std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
subgraph_with_mapper(const at::Tensor& rowptr,
                     const at::Tensor& col,
                     const at::Tensor& nodes,
                     const Mapper<T>& mapper,
                     const bool return_edge_id);

}  // namespace sampler
}  // namespace pyg
