#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace partition {

// Returns cluster indices for each node in the graph given by `(rowptr, col)`.
PYG_API at::Tensor metis(
    const at::Tensor& rowptr,
    const at::Tensor& col,
    int64_t num_partitions,
    const c10::optional<at::Tensor>& node_weight = c10::nullopt,
    const c10::optional<at::Tensor>& edge_weight = c10::nullopt,
    bool recursive = false);

}  // namespace partition
}  // namespace pyg
