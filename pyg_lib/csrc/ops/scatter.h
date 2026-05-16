#pragma once

#include <ATen/ATen.h>
#include <optional>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace ops {

// Sums all values from `src` into `out` at the indices specified in `index`
// along a given axis `dim`.
//
// When `out` is not provided, a fresh zero-initialized buffer is allocated.
// When `out` *is* provided, the operation **accumulates** into the caller's
// buffer (no zero-init); this matches the upstream contract and is what makes
// the symmetric-pair backwards (`gather_coo` / `gather_csr`) efficient.
PYG_API at::Tensor scatter_sum(
    const at::Tensor& src,
    const at::Tensor& index,
    int64_t dim = -1,
    const std::optional<at::Tensor>& out = std::nullopt,
    std::optional<int64_t> dim_size = std::nullopt);

}  // namespace ops
}  // namespace pyg
