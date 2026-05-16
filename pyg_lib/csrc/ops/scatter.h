#pragma once

#include <ATen/ATen.h>
#include <optional>
#include <tuple>
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

// Multiplies all values from `src` into `out` at the indices specified in
// `index` along a given axis `dim`.
//
// When `out` is not provided, a fresh ones-initialized buffer is allocated
// (so that the multiplicative reduction starts from the multiplicative
// identity). When `out` *is* provided, the operation multiplies into the
// caller's buffer (no ones-init); the caller is responsible for any non-
// default starting state. This matches the upstream contract.
PYG_API at::Tensor scatter_mul(
    const at::Tensor& src,
    const at::Tensor& index,
    int64_t dim = -1,
    const std::optional<at::Tensor>& out = std::nullopt,
    std::optional<int64_t> dim_size = std::nullopt);

// Averages all values from `src` into `out` at the indices specified in
// `index` along a given axis `dim`.
//
// Implemented as two `scatter_sum` calls: one to accumulate the per-bucket
// sum, and one (over a ones tensor sized to match `index`) to compute the
// per-bucket count. The final result is the sum divided by the count, with
// empty buckets producing `0` (the count is `masked_fill`-ed up to `1` so
// the division does not generate NaNs). Integer dtypes use floor division
// to match the upstream contract; floating dtypes use true division.
//
// When `out` is provided, the accumulation happens in the caller's buffer
// (no zero-init), matching the upstream `scatter_sum` contract.
PYG_API at::Tensor scatter_mean(
    const at::Tensor& src,
    const at::Tensor& index,
    int64_t dim = -1,
    const std::optional<at::Tensor>& out = std::nullopt,
    std::optional<int64_t> dim_size = std::nullopt);

// Reduces all values from `src` into `out` at the indices specified in
// `index` along a given axis `dim` using a minimum reduction, and returns
// both the per-bucket minimum value and the source position that produced
// it (`arg_out`).
//
// When `out` is not provided, a fresh buffer is allocated and initialized
// to `numeric_limits<scalar_t>::max()` so that the running min is updated
// on the first contributing element. Empty buckets (no contributing source
// element) are reset to `0` after the reduction loop; the matching slot in
// `arg_out` keeps the sentinel value `src.size(dim)` (one past the last
// valid index along `dim`).
//
// When `out` *is* provided, the caller's buffer is used as the running
// state and the caller is responsible for any non-default starting value.
// This matches the upstream `pytorch_scatter` contract.
//
// **CPU determinism:** the CPU kernel produces a *first-match* `arg_out`
// on ties. The CUDA kernel is not guaranteed to match on ties (any valid
// argmin is acceptable).
//
// `arg_out` is non-differentiable; only `out` participates in autograd.
PYG_API std::tuple<at::Tensor, at::Tensor> scatter_min(
    const at::Tensor& src,
    const at::Tensor& index,
    int64_t dim = -1,
    const std::optional<at::Tensor>& out = std::nullopt,
    std::optional<int64_t> dim_size = std::nullopt);

// Reduces all values from `src` into `out` at the indices specified in
// `index` along a given axis `dim` using a maximum reduction, and returns
// both the per-bucket maximum value and the source position that produced
// it (`arg_out`).
//
// When `out` is not provided, a fresh buffer is allocated and initialized
// to `numeric_limits<scalar_t>::lowest()` so that the running max is
// updated on the first contributing element. Empty buckets (no
// contributing source element) are reset to `0` after the reduction loop;
// the matching slot in `arg_out` keeps the sentinel value `src.size(dim)`
// (one past the last valid index along `dim`).
//
// When `out` *is* provided, the caller's buffer is used as the running
// state and the caller is responsible for any non-default starting value.
// This matches the upstream `pytorch_scatter` contract.
//
// **CPU determinism:** the CPU kernel produces a *first-match* `arg_out`
// on ties. The CUDA kernel is not guaranteed to match on ties (any valid
// argmax is acceptable).
//
// `arg_out` is non-differentiable; only `out` participates in autograd.
PYG_API std::tuple<at::Tensor, at::Tensor> scatter_max(
    const at::Tensor& src,
    const at::Tensor& index,
    int64_t dim = -1,
    const std::optional<at::Tensor>& out = std::nullopt,
    std::optional<int64_t> dim_size = std::nullopt);

}  // namespace ops
}  // namespace pyg
