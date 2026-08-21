#pragma once

#include <ATen/ATen.h>
#include <optional>
#include <tuple>
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

// Computes the per-row minimum of `src` along the implicit axis
// `dim = indptr.dim() - 1`, using the compressed row pointer `indptr`. Returns
// `(out, arg_out)` where `arg_out[r]` is the source position that produced the
// minimum value at row `r`.
//
// CSR ops do **not** take a `dim` argument: upstream `pytorch_scatter` fixes
// the reduction axis at `indptr.dim() - 1` (the last indptr dim). They also
// do **not** take a `dim_size`: the output size along `dim` is determined by
// `indptr.size(-1) - 1` (one entry per row, plus a trailing sentinel).
//
// Each row `r` consumes the source slice `src[..., indptr[r]:indptr[r+1], ...]`
// and writes the per-row min to `out[..., r, ...]`. Empty rows
// (`indptr[r+1] == indptr[r]`) are reset to `0` after the reduction loop; the
// matching slot in `arg_out` keeps the sentinel value `src.size(dim)` (one
// past the last valid index along `dim`).
//
// `indptr` is `expand`-broadcast to match `src.shape[:indptr.dim()-1]` along
// the leading dims; we force `.contiguous()` at the kernel boundary.
//
// When `out` is not provided, a fresh buffer is allocated and initialized to
// `numeric_limits<scalar_t>::max()` so that the first contributing element
// always wins. When `out` *is* provided, the caller's buffer is used as the
// running state (no max-init); the caller is responsible for any non-default
// starting value. This matches the upstream `pytorch_scatter` contract.
//
// **CPU determinism:** the CPU kernel produces a *first-match* `arg_out`
// on ties (strict `<` comparison). The CUDA kernel is not guaranteed to
// match on ties (any valid argmin is acceptable).
//
// `arg_out` is non-differentiable; only `out` participates in autograd.
PYG_API std::tuple<at::Tensor, at::Tensor> segment_min_csr(
    const at::Tensor& src,
    const at::Tensor& indptr,
    const std::optional<at::Tensor>& out = std::nullopt);

// Computes the per-row maximum of `src` along the implicit axis
// `dim = indptr.dim() - 1`, using the compressed row pointer `indptr`. Returns
// `(out, arg_out)` where `arg_out[r]` is the source position that produced the
// maximum value at row `r`.
//
// CSR ops do **not** take a `dim` argument: upstream `pytorch_scatter` fixes
// the reduction axis at `indptr.dim() - 1` (the last indptr dim). They also
// do **not** take a `dim_size`: the output size along `dim` is determined by
// `indptr.size(-1) - 1` (one entry per row, plus a trailing sentinel).
//
// Each row `r` consumes the source slice `src[..., indptr[r]:indptr[r+1], ...]`
// and writes the per-row max to `out[..., r, ...]`. Empty rows
// (`indptr[r+1] == indptr[r]`) are reset to `0` after the reduction loop; the
// matching slot in `arg_out` keeps the sentinel value `src.size(dim)` (one
// past the last valid index along `dim`).
//
// `indptr` is `expand`-broadcast to match `src.shape[:indptr.dim()-1]` along
// the leading dims; we force `.contiguous()` at the kernel boundary.
//
// When `out` is not provided, a fresh buffer is allocated and initialized to
// `numeric_limits<scalar_t>::lowest()` so that the first contributing element
// always wins. When `out` *is* provided, the caller's buffer is used as the
// running state (no lowest-init); the caller is responsible for any
// non-default starting value. This matches the upstream `pytorch_scatter`
// contract.
//
// **CPU determinism:** the CPU kernel produces a *first-match* `arg_out`
// on ties (strict `>` comparison). The CUDA kernel is not guaranteed to
// match on ties (any valid argmax is acceptable).
//
// `arg_out` is non-differentiable; only `out` participates in autograd.
PYG_API std::tuple<at::Tensor, at::Tensor> segment_max_csr(
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
