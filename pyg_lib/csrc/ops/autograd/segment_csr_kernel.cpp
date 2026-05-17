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
  m.impl(TORCH_SELECTIVE_NAME("pyg::gather_csr"),
         TORCH_FN(gather_csr_autograd));
}

}  // namespace ops
}  // namespace pyg
