#include "../scatter.h"
#include "../utils.h"

#include <torch/autograd.h>

namespace pyg {
namespace ops {

namespace {

using torch::autograd::variable_list;

// Autograd Function for `pyg::scatter_sum`.
//
// The C++ dispatcher returns a fresh tensor (when `out=None`) or the caller's
// buffer mutated in place (when `out=` is supplied). The `mark_dirty` call in
// the latter case lets gradcheck pick up the in-place mutation; the actual
// backward gradient is computed the same way regardless.
class ScatterSum : public torch::autograd::Function<ScatterSum> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& src,
                               const at::Tensor& index,
                               int64_t dim,
                               const std::optional<at::Tensor>& optional_out,
                               std::optional<int64_t> dim_size) {
    at::AutoDispatchBelowADInplaceOrView g;

    // Normalize `dim` once so saved metadata is consistent with what backward
    // sees from `torch.gather`.
    const int64_t dim_norm = dim < 0 ? src.dim() + dim : dim;

    // Broadcast `index` up to `src.shape` (records unsqueeze/expand on the
    // autograd graph so backward can call `.gather` directly).
    auto index_b = broadcast(index, src, dim_norm);

    auto out = scatter_sum(src, index_b, dim_norm, optional_out, dim_size);

    ctx->save_for_backward({index_b});
    ctx->saved_data["dim"] = dim_norm;

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
    const auto dim = ctx->saved_data["dim"].toInt();

    // `grad_src[i] = grad_out[index[i]]` — exactly `gather` along `dim`.
    auto grad_src = grad_out.gather(dim, index_b);

    // 5 input slots: (src, index, dim, out, dim_size). Only `src` is a real
    // differentiable tensor; the rest get undefined-Tensor grads.
    return {grad_src, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

at::Tensor scatter_sum_autograd(const at::Tensor& src,
                                const at::Tensor& index,
                                int64_t dim,
                                const std::optional<at::Tensor>& optional_out,
                                std::optional<int64_t> dim_size) {
  return ScatterSum::apply(src, index, dim, optional_out, dim_size)[0];
}

// Autograd Function for `pyg::scatter_mul`.
//
// Backward formula (upstream pytorch_scatter `scatter.cpp:101-112`):
//
//   grad_src = (grad_out * out).gather(dim, index_b) / src
//
// where `out` is the forward result. At positions where `src == 0`, this
// expression is ill-defined (0/0 NaN); upstream produces a NaN and then
// masks it with `masked_fill_(isnan, 0)`. We use `torch::where(src != 0,
// ..., zeros)` to produce zeros directly without going through NaN.
class ScatterMul : public torch::autograd::Function<ScatterMul> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& src,
                               const at::Tensor& index,
                               int64_t dim,
                               const std::optional<at::Tensor>& optional_out,
                               std::optional<int64_t> dim_size) {
    at::AutoDispatchBelowADInplaceOrView g;

    const int64_t dim_norm = dim < 0 ? src.dim() + dim : dim;

    // Broadcast `index` up to `src.shape` so that backward's `gather` can
    // use it directly. Recording the broadcast on the autograd graph keeps
    // gradient connectivity (though `index` itself is non-differentiable).
    auto index_b = broadcast(index, src, dim_norm);

    auto out = scatter_mul(src, index_b, dim_norm, optional_out, dim_size);

    // Backward needs `src` (denominator), `index_b` (gather axis), and `out`
    // (so that `grad_out * out` can be gathered at the contributing indices).
    ctx->save_for_backward({src, index_b, out});
    ctx->saved_data["dim"] = dim_norm;

    if (optional_out.has_value()) {
      ctx->mark_dirty({optional_out.value()});
    }

    return {out};
  }

  static variable_list backward(torch::autograd::AutogradContext* ctx,
                                variable_list grad_outs) {
    const auto grad_out = grad_outs[0];
    const auto saved = ctx->get_saved_variables();
    const auto src = saved[0];
    const auto index_b = saved[1];
    const auto out = saved[2];
    const auto dim = ctx->saved_data["dim"].toInt();

    // Note: `(grad_out * out)` is the gradient that arrives at the bucket;
    // gathering it back to the source positions multiplies by the product
    // of all *other* contributors, which is `out / src`. Hence the divide.
    auto gathered = (grad_out * out).gather(dim, index_b);
    auto grad_src = at::where(src != 0, gathered / src, at::zeros_like(src));

    // 5 input slots: (src, index, dim, out, dim_size).
    return {grad_src, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

at::Tensor scatter_mul_autograd(const at::Tensor& src,
                                const at::Tensor& index,
                                int64_t dim,
                                const std::optional<at::Tensor>& optional_out,
                                std::optional<int64_t> dim_size) {
  return ScatterMul::apply(src, index, dim, optional_out, dim_size)[0];
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, Autograd, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::scatter_sum"),
         TORCH_FN(scatter_sum_autograd));
  m.impl(TORCH_SELECTIVE_NAME("pyg::scatter_mul"),
         TORCH_FN(scatter_mul_autograd));
}

}  // namespace ops
}  // namespace pyg
