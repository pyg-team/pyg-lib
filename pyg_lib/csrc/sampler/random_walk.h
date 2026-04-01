#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace sampler {

// Samples random walks of length `walk_length` from all nodes indices in `seed`
// in the graph given by `(rowptr, col)`.
// Returns: a tensor of shape `[seed.size(0), walk_length + 1]`
PYG_API at::Tensor random_walk(const at::Tensor& rowptr,
                               const at::Tensor& col,
                               const at::Tensor& seed,
                               int64_t walk_length,
                               double p = 1.0,
                               double q = 1.0);

}  // namespace sampler
}  // namespace pyg
