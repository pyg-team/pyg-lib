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
  m.impl(TORCH_SELECTIVE_NAME("pyg::gather_coo"),
         TORCH_FN(gather_coo_autograd));
}

}  // namespace ops
}  // namespace pyg
