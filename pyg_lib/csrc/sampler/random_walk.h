#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace sampler {

// Samples random walks of length `walk_length` from all nodes indices in `seed`
// in the graph given by `(rowptr, col)`.
// Returns: a tuple of (node_seq, edge_seq) where
//   node_seq is of shape `[seed.size(0), walk_length + 1]`
//   edge_seq is of shape `[seed.size(0), walk_length]`
PYG_API std::tuple<at::Tensor, at::Tensor> random_walk(const at::Tensor& rowptr,
                                                       const at::Tensor& col,
                                                       const at::Tensor& seed,
                                                       int64_t walk_length,
                                                       double p = 1.0,
                                                       double q = 1.0);

}  // namespace sampler
}  // namespace pyg
