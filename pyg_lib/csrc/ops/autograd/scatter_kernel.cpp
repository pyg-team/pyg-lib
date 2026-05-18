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

}  // namespace

TORCH_LIBRARY_IMPL(pyg, Autograd, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::scatter_sum"),
         TORCH_FN(scatter_sum_autograd));
}

}  // namespace ops
}  // namespace pyg
