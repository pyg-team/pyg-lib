#include "../softmax.h"

#include <torch/autograd.h>

namespace pyg {
namespace ops {

namespace {

using torch::autograd::variable_list;

class SoftmaxCSR : public torch::autograd::Function<SoftmaxCSR> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& src,
                               const at::Tensor& ptr,
                               const int64_t dim) {
    at::AutoDispatchBelowADInplaceOrView g;

    at::Tensor out = softmax_csr(src, ptr, dim);
    ctx->saved_data["dim"] = dim;
    ctx->save_for_backward({src, out, ptr});

    return {out};
  }

  static variable_list backward(torch::autograd::AutogradContext* ctx,
                                variable_list out_grads) {
    const auto out_grad = out_grads[0];
    const auto saved = ctx->get_saved_variables();
    const auto src = saved[0];
    const auto out = saved[1];
    const auto ptr = saved[2];
    const auto dim = ctx->saved_data["dim"].toInt();

    auto src_grad = at::Tensor();
    if (torch::autograd::any_variable_requires_grad({src})) {
      src_grad = softmax_csr_backward(out, out_grad, ptr, dim);
    }

    return {src_grad, at::Tensor(), at::Tensor()};
  }
};

at::Tensor softmax_csr_autograd(const at::Tensor& src,
                                const at::Tensor& ptr,
                                const int64_t dim) {
  return SoftmaxCSR::apply(src, ptr, dim)[0];
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, Autograd, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::softmax_csr"),
         TORCH_FN(softmax_csr_autograd));
}

}  // namespace ops
}  // namespace pyg
