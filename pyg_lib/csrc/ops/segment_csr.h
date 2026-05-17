#pragma once

#include <ATen/ATen.h>
#include <optional>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace ops {

// Sums all values from `src` into `out` at the segments specified by the
// compressed row pointer `indptr` along the implicit axis
// `dim = indptr.dim() - 1`.
//
// CSR ops do **not** take a `dim` argument: upstream `pytorch_scatter` fixes
// the reduction axis at `indptr.dim() - 1` (the last indptr dim). They also
// do **not** take a `dim_size`: the output size along `dim` is determined by
// `indptr.size(-1) - 1` (one entry per row, plus a trailing sentinel).
//
// Each row `r` consumes the source slice `src[..., indptr[r]:indptr[r+1], ...]`
// and writes the per-row sum into `out[..., r, ...]`. Empty rows
// (`indptr[r+1] == indptr[r]`) leave the corresponding `out` slot at its
// zero-initialized value (or unchanged if a caller-supplied `out=` is given).
//
// `indptr` is `expand`-broadcast to match `src.shape[:indptr.dim()-1]` along
// the leading dims; we force `.contiguous()` at the kernel boundary.
//
// When `out` is not provided, a fresh zero-initialized buffer is allocated.
// When `out` *is* provided, the operation **accumulates** into the caller's
// buffer (no zero-init); this matches the upstream contract and is what makes
// `GatherCSR::backward` efficient (the backward allocates
// `at::zeros(src_shape)` and calls `segment_sum_csr` with it as `out=` to
// deposit the gradient in-place).
PYG_API at::Tensor segment_sum_csr(
    const at::Tensor& src,
    const at::Tensor& indptr,
    const std::optional<at::Tensor>& out = std::nullopt);

// Computes the per-row mean of `src` along the implicit axis
// `dim = indptr.dim() - 1`, using the compressed row pointer `indptr`.
//
// CSR ops do **not** take a `dim` argument: upstream `pytorch_scatter` fixes
// the reduction axis at `indptr.dim() - 1` (the last indptr dim). They also
// do **not** take a `dim_size`: the output size along `dim` is determined by
// `indptr.size(-1) - 1` (one entry per row, plus a trailing sentinel).
//
// Each row `r` consumes the source slice `src[..., indptr[r]:indptr[r+1], ...]`
// and writes the per-row mean (sum / row-length) into `out[..., r, ...]`.
// Empty rows (`indptr[r+1] == indptr[r]`) are handled by clamping their
// row-length to 1 before division; the corresponding `out` slot is left at
// its zero value (since the sum loop contributes nothing).
//
// `indptr` is `expand`-broadcast to match `src.shape[:indptr.dim()-1]` along
// the leading dims; we force `.contiguous()` at the kernel boundary.
//
// `out=` contract (matches upstream): mean does **not** accumulate into a
// caller-supplied buffer — it overwrites the per-row slots. We allocate the
// `out` buffer zero-initialized so that empty-row slots stay at 0.
PYG_API at::Tensor segment_mean_csr(
    const at::Tensor& src,
    const at::Tensor& indptr,
    const std::optional<at::Tensor>& out = std::nullopt);

// Gathers values from `src` at the row positions encoded in `indptr`, along
// the implicit axis `dim = indptr.dim() - 1`. Concretely, for each row `r`,
// the value `src[..., r, ...]` is broadcast to all output positions
// `out[..., i, ...]` for `i ∈ [indptr[r], indptr[r+1])`.
//
// CSR ops do **not** take a `dim` argument: upstream `pytorch_scatter` fixes
// the gather axis at `indptr.dim() - 1`. The output size along `dim` is
// determined by the last entry of `indptr` (i.e. `indptr[..., -1]`).
//
// `indptr` is `expand`-broadcast to match `src.shape[:indptr.dim()-1]` along
// the leading dims; we force `.contiguous()` at the kernel boundary.
//
// When `out` is not provided, a fresh buffer is allocated. When `out` *is*
// provided, the operation **overwrites** the caller's buffer per element
// (matches upstream `pytorch_scatter`).
PYG_API at::Tensor gather_csr(
    const at::Tensor& src,
    const at::Tensor& indptr,
    const std::optional<at::Tensor>& out = std::nullopt);

}  // namespace ops
}  // namespace pyg
