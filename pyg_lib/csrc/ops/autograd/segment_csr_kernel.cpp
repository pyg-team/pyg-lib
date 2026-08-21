#include "../segment_csr.h"

#include <torch/autograd.h>

namespace pyg {
namespace ops {

namespace {

using torch::autograd::variable_list;

// Autograd Function for `pyg::segment_sum_csr`.
//
// CSR ops fix the reduction axis at `dim = indptr.dim() - 1`. Forward saves
// `indptr`; backward is exactly `gather_csr(grad_out, indptr)` — the
// symmetric-pair inverse of forward (each `grad_out[r]` is broadcast back to
// every source position `i ∈ [indptr[r], indptr[r+1])`).
//
// `out=` contract: the C++ dispatcher accumulates into the caller's buffer
// when `out=` is supplied. `mark_dirty` lets gradcheck pick up the in-place
// mutation; the backward gradient itself is computed the same way regardless.
class SegmentSumCSR : public torch::autograd::Function<SegmentSumCSR> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& src,
                               const at::Tensor& indptr,
                               const std::optional<at::Tensor>& optional_out) {
    at::AutoDispatchBelowADInplaceOrView g;

    auto out = segment_sum_csr(src, indptr, optional_out);

    // Save `indptr` for backward's `gather_csr` call. The gather kernel
    // applies the same broadcast convention, so we save the *original*
    // (possibly pre-broadcast) `indptr`.
    ctx->save_for_backward({indptr});
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
    const auto indptr = saved[0];
    auto src_shape = ctx->saved_data["src_shape"].toIntList().vec();

    // `grad_src[..., i, ...] = grad_out[..., r, ...]` where `r` is the row
    // such that `i ∈ [indptr[r], indptr[r+1])` — exactly `gather_csr`.
    // We allocate a `grad_in` buffer the size of `src` and let `gather_csr`
    // fill it via the `out=` overwrite contract. Positions in `grad_in` that
    // are not covered by any `[indptr[r], indptr[r+1])` range (e.g. when
    // `src` has trailing entries past `indptr[-1]`) remain uninitialized;
    // upstream `segment_csr.cpp:75-76` uses `torch::empty` here for the same
    // reason — those positions never received any contribution in forward,
    // so their gradient is undefined.
    auto grad_in = at::empty(src_shape, grad_out.options());
    gather_csr(grad_out, indptr, grad_in);

    // 3 input slots: (src, indptr, out). Only `src` is a real differentiable
    // tensor; the rest get undefined-Tensor grads.
    return {grad_in, at::Tensor(), at::Tensor()};
  }
};

at::Tensor segment_sum_csr_autograd(
    const at::Tensor& src,
    const at::Tensor& indptr,
    const std::optional<at::Tensor>& optional_out) {
  return SegmentSumCSR::apply(src, indptr, optional_out)[0];
}

// Autograd Function for `pyg::segment_mean_csr`.
//
// Forward computes the per-row mean of `src` over `[indptr[r], indptr[r+1])`.
// Backward routes each `grad_out[r]` back to every source entry in that row,
// scaled by `1 / row_length[r]`. We use `gather_csr(grad_out, indptr)` to
// broadcast `grad_out` back to source positions, and `gather_csr(count, ...)`
// to lift the per-row count to per-source positions; then in-place divide.
//
// Row-length recompute: cheap (just `indptr.narrow().diff()`) and avoids
// saving an extra tensor for backward (the COO mean kernel has to save the
// count because reconstructing it from `index` would require a second pass).
//
// Empty-tensor guard: if `grad_in.numel() == 0` (e.g. `src` was empty), the
// gather/divide both no-op and we can skip them — matches upstream
// `segment_csr.cpp:100-109`.
class SegmentMeanCSR : public torch::autograd::Function<SegmentMeanCSR> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& src,
                               const at::Tensor& indptr,
                               const std::optional<at::Tensor>& optional_out) {
    at::AutoDispatchBelowADInplaceOrView g;

    auto out = segment_mean_csr(src, indptr, optional_out);

    ctx->save_for_backward({indptr});
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
    const auto indptr = saved[0];
    auto src_shape = ctx->saved_data["src_shape"].toIntList().vec();

    auto grad_in = at::empty(src_shape, grad_out.options());

    // Match upstream `segment_csr.cpp:100-109`: skip gather/divide when the
    // gradient buffer is empty (e.g. `src` was empty); the empty tensor
    // itself is the correct gradient.
    if (grad_in.numel() > 0) {
      // Lift `grad_out` to per-source positions via the CSR-symmetric
      // inverse of forward (uses the `out=` overwrite contract of
      // `gather_csr` to fill `grad_in` in place).
      gather_csr(grad_out, indptr, grad_in);

      // Per-row count from `indptr`: `count[r] = indptr[r+1] - indptr[r]`.
      // `indptr` may carry leading dims (when broadcast); narrowing the
      // last dim produces a `count` with shape `indptr.shape[:-1] + (R,)`
      // where `R = indptr.size(-1) - 1`. Cast to grad dtype so the divide
      // is in-place safe.
      auto indptr1 = indptr.narrow(-1, 0, indptr.size(-1) - 1);
      auto indptr2 = indptr.narrow(-1, 1, indptr.size(-1) - 1);
      auto count = (indptr2 - indptr1).to(grad_in.options());
      // `gather_csr` lifts the per-row count to per-source positions along
      // the CSR axis. Unsqueeze the result over trailing K dims of
      // `grad_out` past `indptr.dim()` so the divide broadcasts correctly.
      count = gather_csr(count, indptr, std::nullopt);
      for (int64_t i = 0; i < grad_out.dim() - indptr.dim(); ++i)
        count = count.unsqueeze(-1);
      grad_in.true_divide_(count);
    }

    // 3 input slots: (src, indptr, out). Only `src` is a real differentiable
    // tensor; the rest get undefined-Tensor grads.
    return {grad_in, at::Tensor(), at::Tensor()};
  }
};

at::Tensor segment_mean_csr_autograd(
    const at::Tensor& src,
    const at::Tensor& indptr,
    const std::optional<at::Tensor>& optional_out) {
  return SegmentMeanCSR::apply(src, indptr, optional_out)[0];
}

// Autograd Function for `pyg::segment_min_csr`.
//
// CSR ops fix the reduction axis at `dim = indptr.dim() - 1`. Forward returns
// `(out, arg_out)`. Only `out` is differentiable; `arg_out` is marked
// non-differentiable so PyTorch will not error when callers try to take
// gradients through it (mirrors `SegmentMinCOO` / `ScatterMin`).
//
// Backward formula (mirrors `ScatterMin` from commit 4):
//
//   src_shape_plus = src_shape; src_shape_plus[dim] += 1
//   grad_in = zeros(src_shape_plus)
//   grad_in.scatter_(dim, arg_out, grad_out)   // sentinel lands in +1 slot
//   grad_in = grad_in.narrow(dim, 0, src_shape[dim])  // drop sentinel
//
// The `+1`/`narrow` trick routes gradient only to the source positions that
// produced the per-row min and silently drops any contribution from rows
// whose `arg_out` is still at the sentinel (empty rows).
class SegmentMinCSR : public torch::autograd::Function<SegmentMinCSR> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& src,
                               const at::Tensor& indptr,
                               const std::optional<at::Tensor>& optional_out) {
    at::AutoDispatchBelowADInplaceOrView g;

    const int64_t dim = indptr.dim() - 1;
    TORCH_CHECK(dim >= 0,
                "segment_min_csr: indptr must have at least 1 dimension");

    auto result = segment_min_csr(src, indptr, optional_out);
    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result);

    // Backward needs `arg_out` (where to deposit each `grad_out`), the
    // implicit `dim`, and the original `src.sizes()` (so the +1/narrow trick
    // reconstructs the right shape). We save `indptr` for parity with other
    // CSR Functions; it is not used directly by backward.
    ctx->save_for_backward({indptr, arg_out});
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
    // saved[0] is `indptr`, kept for parity with other CSR Functions; not
    // used directly by backward (the +1/narrow trick references `arg_out`).
    const auto arg_out = saved[1];
    const auto dim = ctx->saved_data["dim"].toInt();
    auto src_shape = ctx->saved_data["src_shape"].toIntList().vec();

    // `+1`/`narrow` trick: extend `src_shape` along `dim` by one extra
    // sentinel slot, scatter `grad_out` into the extended buffer using
    // `arg_out` (sentinel entries land in the trailing slot and are dropped
    // by the subsequent `narrow`).
    src_shape[dim] += 1;
    auto grad_in = at::zeros(src_shape, grad_out.options());
    grad_in.scatter_(dim, arg_out, grad_out);
    grad_in = grad_in.narrow(dim, 0, src_shape[dim] - 1);

    // 3 input slots: (src, indptr, out). Only `src` is a real differentiable
    // tensor; the rest get undefined-Tensor grads.
    return {grad_in, at::Tensor(), at::Tensor()};
  }
};

std::tuple<at::Tensor, at::Tensor> segment_min_csr_autograd(
    const at::Tensor& src,
    const at::Tensor& indptr,
    const std::optional<at::Tensor>& optional_out) {
  auto result = SegmentMinCSR::apply(src, indptr, optional_out);
  return std::make_tuple(result[0], result[1]);
}

// Autograd Function for `pyg::segment_max_csr`.
//
// Symmetric to `SegmentMinCSR`: forward returns `(out, arg_out)`, only `out`
// is differentiable. Backward uses the same `+1`/`narrow` trick to route
// gradient only to the source positions that produced the per-row max and
// silently drop sentinel slots (empty rows).
class SegmentMaxCSR : public torch::autograd::Function<SegmentMaxCSR> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& src,
                               const at::Tensor& indptr,
                               const std::optional<at::Tensor>& optional_out) {
    at::AutoDispatchBelowADInplaceOrView g;

    const int64_t dim = indptr.dim() - 1;
    TORCH_CHECK(dim >= 0,
                "segment_max_csr: indptr must have at least 1 dimension");

    auto result = segment_max_csr(src, indptr, optional_out);
    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result);

    ctx->save_for_backward({indptr, arg_out});
    ctx->saved_data["dim"] = dim;
    ctx->saved_data["src_shape"] = src.sizes();

    ctx->mark_non_differentiable({arg_out});

    if (optional_out.has_value()) {
      ctx->mark_dirty({optional_out.value()});
    }

    return {out, arg_out};
  }

  static variable_list backward(torch::autograd::AutogradContext* ctx,
                                variable_list grad_outs) {
    const auto grad_out = grad_outs[0];
    const auto saved = ctx->get_saved_variables();
    const auto arg_out = saved[1];
    const auto dim = ctx->saved_data["dim"].toInt();
    auto src_shape = ctx->saved_data["src_shape"].toIntList().vec();

    src_shape[dim] += 1;
    auto grad_in = at::zeros(src_shape, grad_out.options());
    grad_in.scatter_(dim, arg_out, grad_out);
    grad_in = grad_in.narrow(dim, 0, src_shape[dim] - 1);

    return {grad_in, at::Tensor(), at::Tensor()};
  }
};

std::tuple<at::Tensor, at::Tensor> segment_max_csr_autograd(
    const at::Tensor& src,
    const at::Tensor& indptr,
    const std::optional<at::Tensor>& optional_out) {
  auto result = SegmentMaxCSR::apply(src, indptr, optional_out);
  return std::make_tuple(result[0], result[1]);
}

// Autograd Function for `pyg::gather_csr`.
//
// CSR ops fix the gather axis at `dim = indptr.dim() - 1`. Forward saves
// `indptr` and `src.sizes()`; backward allocates `at::zeros(src_shape)` and
// calls `segment_sum_csr(grad_out, indptr, /*out=*/grad_in)`, which
// accumulates each `grad_out` entry into its source position via the `out=`
// accumulate contract. This is the inverse of forward.
class GatherCSR : public torch::autograd::Function<GatherCSR> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& src,
                               const at::Tensor& indptr,
                               const std::optional<at::Tensor>& optional_out) {
    at::AutoDispatchBelowADInplaceOrView g;

    auto out = gather_csr(src, indptr, optional_out);

    // Backward needs `indptr` (to know which `grad_out` entries map to which
    // source row) and `src.sizes()` (to allocate the right-shaped `grad_in`).
    ctx->save_for_backward({indptr});
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
    const auto indptr = saved[0];
    auto src_shape = ctx->saved_data["src_shape"].toIntList().vec();

    // Allocate the gradient buffer at `src.shape` and use the `out=`
    // accumulate contract of `segment_sum_csr` to deposit each `grad_out`
    // entry at its source row. This is the inverse of forward: each
    // `grad_in[r]` accumulates `grad_out[i]` for all `i ∈ [indptr[r],
    // indptr[r+1])`.
    auto grad_in = at::zeros(src_shape, grad_out.options());
    segment_sum_csr(grad_out, indptr, /*out=*/grad_in);

    // 3 input slots: (src, indptr, out). Only `src` is a real differentiable
    // tensor; the rest get undefined-Tensor grads.
    return {grad_in, at::Tensor(), at::Tensor()};
  }
};

at::Tensor gather_csr_autograd(const at::Tensor& src,
                               const at::Tensor& indptr,
                               const std::optional<at::Tensor>& optional_out) {
  return GatherCSR::apply(src, indptr, optional_out)[0];
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, Autograd, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_sum_csr"),
         TORCH_FN(segment_sum_csr_autograd));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_mean_csr"),
         TORCH_FN(segment_mean_csr_autograd));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_min_csr"),
         TORCH_FN(segment_min_csr_autograd));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_max_csr"),
         TORCH_FN(segment_max_csr_autograd));
  m.impl(TORCH_SELECTIVE_NAME("pyg::gather_csr"),
         TORCH_FN(gather_csr_autograd));
}

}  // namespace ops
}  // namespace pyg
