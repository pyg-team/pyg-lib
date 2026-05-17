#pragma once

#include <ATen/ATen.h>
#include <optional>
#include <tuple>
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

// Computes the mean of all values from `src` at the segment positions
// specified in `index` along the implicit axis `dim = index.dim() - 1`. The
// per-bucket count is computed internally during the sequential pass and
// applied as a final division.
//
// COO ops do **not** take a `dim` argument: upstream `pytorch_scatter` fixes
// the reduction axis at `index.dim() - 1`. The `index` tensor must be
// **sorted ascending** along that axis.
//
// `out=` contract: unlike `segment_sum_coo` (which accumulates), the mean
// kernel **overwrites** `out` at every bucket touched by `index` (matches
// upstream `segment_coo_cpu.cpp` for the MEAN reducer). Buckets not visited
// by any `index` entry are left at their original value when `out=` is
// supplied, or zero-initialized otherwise.
PYG_API at::Tensor segment_mean_coo(
    const at::Tensor& src,
    const at::Tensor& index,
    const std::optional<at::Tensor>& out = std::nullopt,
    std::optional<int64_t> dim_size = std::nullopt);

// Reduces all values from `src` into `out` at the segment positions specified
// in `index` along the implicit axis `dim = index.dim() - 1` using a minimum
// reduction, and returns both the per-bucket minimum value and the source
// position that produced it (`arg_out`).
//
// COO ops do **not** take a `dim` argument: upstream `pytorch_scatter` fixes
// the reduction axis at `index.dim() - 1`. The `index` tensor must be
// **sorted ascending** along that axis.
//
// When `out` is not provided, a fresh buffer is allocated and initialized
// to `numeric_limits<scalar_t>::max()` so that the running min is updated
// on the first contributing element. Empty buckets (no contributing source
// element) are reset to `0` after the reduction loop; the matching slot in
// `arg_out` keeps the sentinel value `src.size(dim)` (one past the last
// valid index along `dim`).
//
// When `out` *is* provided, the caller's buffer is used as the running
// state (no max-init); the caller is responsible for any non-default
// starting value. This matches the upstream `pytorch_scatter` contract.
//
// **CPU determinism:** the CPU kernel produces a *first-match* `arg_out`
// on ties (strict `<` comparison). The CUDA kernel is not guaranteed to
// match on ties (any valid argmin is acceptable).
//
// `arg_out` is non-differentiable; only `out` participates in autograd.
PYG_API std::tuple<at::Tensor, at::Tensor> segment_min_coo(
    const at::Tensor& src,
    const at::Tensor& index,
    const std::optional<at::Tensor>& out = std::nullopt,
    std::optional<int64_t> dim_size = std::nullopt);

// Reduces all values from `src` into `out` at the segment positions specified
// in `index` along the implicit axis `dim = index.dim() - 1` using a maximum
// reduction, and returns both the per-bucket maximum value and the source
// position that produced it (`arg_out`).
//
// COO ops do **not** take a `dim` argument: upstream `pytorch_scatter` fixes
// the reduction axis at `index.dim() - 1`. The `index` tensor must be
// **sorted ascending** along that axis.
//
// When `out` is not provided, a fresh buffer is allocated and initialized
// to `numeric_limits<scalar_t>::lowest()` so that the running max is updated
// on the first contributing element. Empty buckets (no contributing source
// element) are reset to `0` after the reduction loop; the matching slot in
// `arg_out` keeps the sentinel value `src.size(dim)` (one past the last
// valid index along `dim`).
//
// When `out` *is* provided, the caller's buffer is used as the running
// state (no lowest-init); the caller is responsible for any non-default
// starting value. This matches the upstream `pytorch_scatter` contract.
//
// **CPU determinism:** the CPU kernel produces a *first-match* `arg_out`
// on ties (strict `>` comparison). The CUDA kernel is not guaranteed to
// match on ties (any valid argmax is acceptable).
//
// `arg_out` is non-differentiable; only `out` participates in autograd.
PYG_API std::tuple<at::Tensor, at::Tensor> segment_max_coo(
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
