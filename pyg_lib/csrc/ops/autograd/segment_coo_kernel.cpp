#include "../segment_coo.h"

#include <torch/autograd.h>

namespace pyg {
namespace ops {

namespace {

using torch::autograd::variable_list;

// Autograd Function for `pyg::segment_sum_coo`.
//
// COO ops fix the reduction axis at `dim = index.dim() - 1`. Forward saves
// `index` (the kernel handles its own `expand`); backward is exactly
// `gather_coo(grad_out, index)` — the symmetric-pair inverse of forward.
//
// `out=` contract: the C++ dispatcher accumulates into the caller's buffer
// when `out=` is supplied. `mark_dirty` lets gradcheck pick up the in-place
// mutation; the backward gradient itself is computed the same way regardless.
class SegmentSumCOO : public torch::autograd::Function<SegmentSumCOO> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& src,
                               const at::Tensor& index,
                               const std::optional<at::Tensor>& optional_out,
                               std::optional<int64_t> dim_size) {
    at::AutoDispatchBelowADInplaceOrView g;

    auto out = segment_sum_coo(src, index, optional_out, dim_size);

    // Save `index` for backward's `gather_coo` call. The gather kernel
    // applies the same `expand` convention, so we save the *original*
    // (possibly pre-broadcast) `index`.
    ctx->save_for_backward({index});

    if (optional_out.has_value()) {
      ctx->mark_dirty({optional_out.value()});
    }

    return {out};
  }

  static variable_list backward(torch::autograd::AutogradContext* ctx,
                                variable_list grad_outs) {
    const auto grad_out = grad_outs[0];
    const auto saved = ctx->get_saved_variables();
    const auto index = saved[0];

    // `grad_src[..., e, ...] = grad_out[..., index[..., e], ...]` — exactly
    // what `gather_coo` computes along `dim = index.dim() - 1`.
    auto grad_src = gather_coo(grad_out, index, std::nullopt);

    // 4 input slots: (src, index, out, dim_size). Only `src` is a real
    // differentiable tensor; the rest get undefined-Tensor grads.
    return {grad_src, at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

at::Tensor segment_sum_coo_autograd(
    const at::Tensor& src,
    const at::Tensor& index,
    const std::optional<at::Tensor>& optional_out,
    std::optional<int64_t> dim_size) {
  return SegmentSumCOO::apply(src, index, optional_out, dim_size)[0];
}

// Autograd Function for `pyg::segment_mean_coo`.
//
// Forward computes the per-bucket mean of `src` at the segment positions
// in `index`. Backward routes each `grad_out[bucket]` back to every source
// entry that landed in that bucket, scaled by `1 / count[bucket]`. We do
// this with the symmetric `gather_coo` (which lifts a per-bucket value to
// per-entry positions) of `(grad_out / count)`.
//
// Storage: we save `index_b` (the post-expand index, contiguous) and a
// freshly recomputed flat `count` of shape `index_b.sizes()` with the last
// dim replaced by `N`. Recomputing count at backward time would require
// re-running the sequential pass; saving it keeps backward O(E + N) instead
// of O(2E + N). The count is stored flat (no trailing K dims); backward
// `gather_coo`s it to per-entry positions and unsqueezes for the K dims.
class SegmentMeanCOO : public torch::autograd::Function<SegmentMeanCOO> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& src,
                               const at::Tensor& index,
                               const std::optional<at::Tensor>& optional_out,
                               std::optional<int64_t> dim_size) {
    at::AutoDispatchBelowADInplaceOrView g;

    // Mirror the kernel's broadcast+contiguous so backward sees an index of
    // shape `src.shape[:index.dim()]`. Also compute the count tensor here
    // for backward; we run the (cheap) sum-coo-style accumulate-and-count
    // pass via existing primitives: build a per-entry "ones" tensor along
    // the dim axis and segment_sum it. That keeps the kernel itself focused
    // on the mean compute path.
    const int64_t dim = index.dim() - 1;
    TORCH_CHECK(dim >= 0,
                "segment_mean_coo: index must have at least 1 dimension");

    auto sizes = index.sizes().vec();
    for (int64_t i = 0; i < index.dim(); ++i)
      sizes[i] = src.size(i);
    auto index_b = index.expand(sizes).contiguous();

    auto out = segment_mean_coo(src, index, optional_out, dim_size);

    // Build the flat count tensor with shape `index_b.shape` but last dim
    // replaced by `out.size(dim)`. We compute it by segment-summing a
    // `ones`-tensor shaped like `index_b` along the dim axis.
    auto ones = at::ones(index_b.sizes(), out.options());
    auto count = segment_sum_coo(ones, index_b, std::nullopt, out.size(dim));

    ctx->save_for_backward({index_b, count});
    ctx->saved_data["src_shape"] = src.sizes();

    if (optional_out.has_value()) {
      ctx->mark_dirty({optional_out.value()});
    }

    return {out};
  }

  static variable_list backward(torch::autograd::AutogradContext* ctx,
                                variable_list grad_outs) {
    const auto grad_out = grad_outs[0];
    const auto saved = ctx->get_saved_variables();
    const auto index_b = saved[0];
    auto count = saved[1];
    auto src_shape = ctx->saved_data["src_shape"].toIntList().vec();

    auto grad_in = at::empty(src_shape, grad_out.options());

    // Match upstream `segment_csr.cpp:100-109`: skip the gather/divide when
    // `grad_in` is empty (e.g. empty `src`); the empty tensor itself is the
    // correct gradient.
    if (grad_in.numel() > 0) {
      gather_coo(grad_out, index_b, grad_in);

      // Lift the per-bucket count to per-entry positions along `dim`, then
      // broadcast over the trailing K dims by unsqueezing.
      count = gather_coo(count, index_b, std::nullopt);
      for (int64_t i = 0; i < grad_out.dim() - index_b.dim(); ++i)
        count = count.unsqueeze(-1);
      grad_in.true_divide_(count);
    }

    // 4 input slots: (src, index, out, dim_size). Only `src` is real.
    return {grad_in, at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

at::Tensor segment_mean_coo_autograd(
    const at::Tensor& src,
    const at::Tensor& index,
    const std::optional<at::Tensor>& optional_out,
    std::optional<int64_t> dim_size) {
  return SegmentMeanCOO::apply(src, index, optional_out, dim_size)[0];
}

// Autograd Function for `pyg::segment_min_coo`.
//
// COO ops fix the reduction axis at `dim = index.dim() - 1`. Forward returns
// `(out, arg_out)`. Only `out` is differentiable; `arg_out` is marked
// non-differentiable so PyTorch will not error when callers try to take
// gradients through it (mirrors `ScatterMin`).
//
// Backward formula (mirrors `ScatterMin` from commit 4):
//
//   src_shape_plus = src_shape; src_shape_plus[dim] += 1
//   grad_in = zeros(src_shape_plus)
//   grad_in.scatter_(dim, arg_out, grad_out)   // sentinel lands in +1 slot
//   grad_in = grad_in.narrow(dim, 0, src_shape[dim])  // drop sentinel
//
// The `+1`/`narrow` trick routes gradient only to the source positions
// that produced the per-bucket min and silently drops any contribution
// from buckets whose `arg_out` is still at the sentinel (empty buckets).
//
// Adapted for COO: `dim = index.dim() - 1`. `arg_out` indexes positions
// along that same `dim` (the reduction axis, which is the last axis of the
// broadcast `index_b`). Because `arg_out` is shaped like `out` (which has
// trailing K dims from `src` that `index_b` does not), we must broadcast
// `arg_out` over the trailing K axes before the `scatter_` call so that
// `grad_in.scatter_(dim, arg_out_b, grad_out)` lines up per (B, N, K).
class SegmentMinCOO : public torch::autograd::Function<SegmentMinCOO> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& src,
                               const at::Tensor& index,
                               const std::optional<at::Tensor>& optional_out,
                               std::optional<int64_t> dim_size) {
    at::AutoDispatchBelowADInplaceOrView g;

    const int64_t dim = index.dim() - 1;
    TORCH_CHECK(dim >= 0,
                "segment_min_coo: index must have at least 1 dimension");

    // Mirror the kernel's broadcast+contiguous so backward sees an index of
    // shape `src.shape[:index.dim()]`. We save `index_b` not because backward
    // uses it directly (only `arg_out` and `src_shape` are needed), but for
    // consistency with the rest of the family and to make any future
    // backward refactor (e.g. via `gather_coo`) trivial.
    auto sizes = index.sizes().vec();
    for (int64_t i = 0; i < index.dim(); ++i)
      sizes[i] = src.size(i);
    auto index_b = index.expand(sizes).contiguous();

    auto result = segment_min_coo(src, index, optional_out, dim_size);
    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result);

    // Backward needs `arg_out` (where to deposit each `grad_out`), the
    // normalized `dim`, and the original `src.sizes()` (so the +1/narrow
    // trick reconstructs the right shape).
    ctx->save_for_backward({index_b, arg_out});
    ctx->saved_data["dim"] = dim;
    ctx->saved_data["src_shape"] = src.sizes();

    // `arg_out` is integer / non-differentiable; mark it so PyTorch will
    // not attempt to propagate gradients through it.
    ctx->mark_non_differentiable({arg_out});

    if (optional_out.has_value()) {
      ctx->mark_dirty({optional_out.value()});
    }

    return {out, arg_out};
  }

  static variable_list backward(torch::autograd::AutogradContext* ctx,
                                variable_list grad_outs) {
    // `grad_outs[0]` is the gradient w.r.t. `out`; `grad_outs[1]` is the
    // gradient w.r.t. `arg_out` and is ignored (non-differentiable).
    const auto grad_out = grad_outs[0];
    const auto saved = ctx->get_saved_variables();
    // saved[0] is `index_b`, kept for parity with other COO Functions; not
    // used directly by backward (the +1/narrow trick references `arg_out`).
    const auto arg_out = saved[1];
    const auto dim = ctx->saved_data["dim"].toInt();
    auto src_shape = ctx->saved_data["src_shape"].toIntList().vec();

    // `+1`/`narrow` trick: extend `src_shape` along `dim` by one extra
    // sentinel slot, scatter `grad_out` into the extended buffer using
    // `arg_out` (sentinel entries land in the trailing slot and are
    // dropped by the subsequent `narrow`).
    src_shape[dim] += 1;
    auto grad_in = at::zeros(src_shape, grad_out.options());
    grad_in.scatter_(dim, arg_out, grad_out);
    grad_in = grad_in.narrow(dim, 0, src_shape[dim] - 1);

    // 4 input slots: (src, index, out, dim_size). Only `src` is a real
    // differentiable tensor; the rest get undefined-Tensor grads.
    return {grad_in, at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

std::tuple<at::Tensor, at::Tensor> segment_min_coo_autograd(
    const at::Tensor& src,
    const at::Tensor& index,
    const std::optional<at::Tensor>& optional_out,
    std::optional<int64_t> dim_size) {
  auto result = SegmentMinCOO::apply(src, index, optional_out, dim_size);
  return std::make_tuple(result[0], result[1]);
}

// Autograd Function for `pyg::segment_max_coo`.
//
// COO ops fix the reduction axis at `dim = index.dim() - 1`. Forward returns
// `(out, arg_out)`. Only `out` is differentiable; `arg_out` is marked
// non-differentiable so PyTorch will not error when callers try to take
// gradients through it (mirrors `ScatterMax`).
//
// Backward formula (mirrors `ScatterMax` from commit 4):
//
//   src_shape_plus = src_shape; src_shape_plus[dim] += 1
//   grad_in = zeros(src_shape_plus)
//   grad_in.scatter_(dim, arg_out, grad_out)   // sentinel lands in +1 slot
//   grad_in = grad_in.narrow(dim, 0, src_shape[dim])  // drop sentinel
//
// The `+1`/`narrow` trick routes gradient only to the source positions
// that produced the per-bucket max and silently drops any contribution
// from buckets whose `arg_out` is still at the sentinel (empty buckets).
//
// Adapted for COO: `dim = index.dim() - 1`. `arg_out` indexes positions
// along that same `dim` (the reduction axis, which is the last axis of the
// broadcast `index_b`). Because `arg_out` is shaped like `out` (which has
// trailing K dims from `src` that `index_b` does not), we must broadcast
// `arg_out` over the trailing K axes before the `scatter_` call so that
// `grad_in.scatter_(dim, arg_out_b, grad_out)` lines up per (B, N, K).
class SegmentMaxCOO : public torch::autograd::Function<SegmentMaxCOO> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& src,
                               const at::Tensor& index,
                               const std::optional<at::Tensor>& optional_out,
                               std::optional<int64_t> dim_size) {
    at::AutoDispatchBelowADInplaceOrView g;

    const int64_t dim = index.dim() - 1;
    TORCH_CHECK(dim >= 0,
                "segment_max_coo: index must have at least 1 dimension");

    // Mirror the kernel's broadcast+contiguous so backward sees an index of
    // shape `src.shape[:index.dim()]`. We save `index_b` not because backward
    // uses it directly (only `arg_out` and `src_shape` are needed), but for
    // consistency with the rest of the family and to make any future
    // backward refactor (e.g. via `gather_coo`) trivial.
    auto sizes = index.sizes().vec();
    for (int64_t i = 0; i < index.dim(); ++i)
      sizes[i] = src.size(i);
    auto index_b = index.expand(sizes).contiguous();

    auto result = segment_max_coo(src, index, optional_out, dim_size);
    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result);

    // Backward needs `arg_out` (where to deposit each `grad_out`), the
    // normalized `dim`, and the original `src.sizes()` (so the +1/narrow
    // trick reconstructs the right shape).
    ctx->save_for_backward({index_b, arg_out});
    ctx->saved_data["dim"] = dim;
    ctx->saved_data["src_shape"] = src.sizes();

    // `arg_out` is integer / non-differentiable; mark it so PyTorch will
    // not attempt to propagate gradients through it.
    ctx->mark_non_differentiable({arg_out});

    if (optional_out.has_value()) {
      ctx->mark_dirty({optional_out.value()});
    }

    return {out, arg_out};
  }

  static variable_list backward(torch::autograd::AutogradContext* ctx,
                                variable_list grad_outs) {
    // `grad_outs[0]` is the gradient w.r.t. `out`; `grad_outs[1]` is the
    // gradient w.r.t. `arg_out` and is ignored (non-differentiable).
    const auto grad_out = grad_outs[0];
    const auto saved = ctx->get_saved_variables();
    // saved[0] is `index_b`, kept for parity with other COO Functions; not
    // used directly by backward (the +1/narrow trick references `arg_out`).
    const auto arg_out = saved[1];
    const auto dim = ctx->saved_data["dim"].toInt();
    auto src_shape = ctx->saved_data["src_shape"].toIntList().vec();

    // `+1`/`narrow` trick: extend `src_shape` along `dim` by one extra
    // sentinel slot, scatter `grad_out` into the extended buffer using
    // `arg_out` (sentinel entries land in the trailing slot and are
    // dropped by the subsequent `narrow`).
    src_shape[dim] += 1;
    auto grad_in = at::zeros(src_shape, grad_out.options());
    grad_in.scatter_(dim, arg_out, grad_out);
    grad_in = grad_in.narrow(dim, 0, src_shape[dim] - 1);

    // 4 input slots: (src, index, out, dim_size). Only `src` is a real
    // differentiable tensor; the rest get undefined-Tensor grads.
    return {grad_in, at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

std::tuple<at::Tensor, at::Tensor> segment_max_coo_autograd(
    const at::Tensor& src,
    const at::Tensor& index,
    const std::optional<at::Tensor>& optional_out,
    std::optional<int64_t> dim_size) {
  auto result = SegmentMaxCOO::apply(src, index, optional_out, dim_size);
  return std::make_tuple(result[0], result[1]);
}

// Autograd Function for `pyg::gather_coo`.
//
// COO ops fix the gather axis at `dim = index.dim() - 1`. Forward saves
// `index` and `src.size(dim)`; backward is exactly `segment_sum_coo` with
// the saved `dim_size`, deposited into a freshly allocated `zeros(src_shape)`
// buffer via the `out=` accumulate contract.
class GatherCOO : public torch::autograd::Function<GatherCOO> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& src,
                               const at::Tensor& index,
                               const std::optional<at::Tensor>& optional_out) {
    at::AutoDispatchBelowADInplaceOrView g;

    auto out = gather_coo(src, index, optional_out);

    // Backward needs `index` (where each output position pulled from), the
    // original `src.sizes()` (to allocate the right-shaped `grad_in`), and
    // `src.size(dim)` (passed as `dim_size` to `segment_sum_coo`).
    const int64_t dim = index.dim() - 1;
    ctx->save_for_backward({index});
    ctx->saved_data["src_shape"] = src.sizes();
    ctx->saved_data["src_size_dim"] = src.size(dim);

    if (optional_out.has_value()) {
      ctx->mark_dirty({optional_out.value()});
    }

    return {out};
  }

  static variable_list backward(torch::autograd::AutogradContext* ctx,
                                variable_list grad_outs) {
    const auto grad_out = grad_outs[0];
    const auto saved = ctx->get_saved_variables();
    const auto index = saved[0];
    auto src_shape = ctx->saved_data["src_shape"].toIntList().vec();
    const int64_t src_size_dim = ctx->saved_data["src_size_dim"].toInt();

    // Allocate the gradient buffer at `src.shape` and use the `out=`
    // accumulate contract of `segment_sum_coo` to deposit each `grad_out`
    // entry at its source position. This is the inverse of forward.
    auto grad_in = at::zeros(src_shape, grad_out.options());
    segment_sum_coo(grad_out, index, /*out=*/grad_in,
                    /*dim_size=*/src_size_dim);

    // 3 input slots: (src, index, out). Only `src` is a real differentiable
    // tensor; the rest get undefined-Tensor grads.
    return {grad_in, at::Tensor(), at::Tensor()};
  }
};

at::Tensor gather_coo_autograd(const at::Tensor& src,
                               const at::Tensor& index,
                               const std::optional<at::Tensor>& optional_out) {
  return GatherCOO::apply(src, index, optional_out)[0];
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, Autograd, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_sum_coo"),
         TORCH_FN(segment_sum_coo_autograd));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_mean_coo"),
         TORCH_FN(segment_mean_coo_autograd));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_min_coo"),
         TORCH_FN(segment_min_coo_autograd));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_max_coo"),
         TORCH_FN(segment_max_coo_autograd));
  m.impl(TORCH_SELECTIVE_NAME("pyg::gather_coo"),
         TORCH_FN(gather_coo_autograd));
}

}  // namespace ops
}  // namespace pyg
