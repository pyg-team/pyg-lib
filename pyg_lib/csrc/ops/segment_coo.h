#pragma once

#include <ATen/ATen.h>
#include <optional>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace ops {

// Sums all values from `src` into `out` at the segment positions specified in
// `index` along the implicit axis `dim = index.dim() - 1`.
//
// COO ops do **not** take a `dim` argument: upstream `pytorch_scatter`
// fixes the reduction axis at `index.dim() - 1` (the last index dim). The
// `index` tensor must be **sorted ascending** along that axis; unsorted input
// is undefined behavior.
//
// When `out` is not provided, a fresh zero-initialized buffer is allocated.
// When `out` *is* provided, the operation **accumulates** into the caller's
// buffer (no zero-init); this matches the upstream contract and is what makes
// the symmetric-pair backward of `gather_coo` efficient (the backward
// allocates `at::zeros(src_shape)` and calls `segment_sum_coo` with it as
// `out=` to deposit the gradient in-place).
PYG_API at::Tensor segment_sum_coo(
    const at::Tensor& src,
    const at::Tensor& index,
    const std::optional<at::Tensor>& out = std::nullopt,
    std::optional<int64_t> dim_size = std::nullopt);

// Gathers values from `src` at positions specified in `index`, along the
// implicit axis `dim = index.dim() - 1`. Concretely:
//
//   out[..., i, ...] = src[..., index[..., i], ...]   along `dim`
//
// COO ops do **not** take a `dim` argument: upstream `pytorch_scatter` fixes
// the gather axis at `index.dim() - 1`.
//
// When `out` is not provided, a fresh buffer is allocated. When `out` *is*
// provided, the operation **overwrites** the caller's buffer per element
// (matches upstream `pytorch_scatter`).
PYG_API at::Tensor gather_coo(
    const at::Tensor& src,
    const at::Tensor& index,
    const std::optional<at::Tensor>& out = std::nullopt);

}  // namespace ops
}  // namespace pyg
