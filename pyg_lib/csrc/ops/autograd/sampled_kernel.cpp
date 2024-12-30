#include "../sampled.h"

#include <torch/autograd.h>

namespace pyg {
namespace ops {

namespace {

using torch::autograd::variable_list;

class SampledOp : public torch::autograd::Function<SampledOp> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& left,
                               const at::Tensor& right,
                               const at::optional<at::Tensor> left_index,
                               const at::optional<at::Tensor> right_index,
                               const std::string fn) {
    at::AutoDispatchBelowADInplaceOrView g;
    at::Tensor out = sampled_op(left, right, left_index, right_index, fn);
    ctx->saved_data["has_left_index"] = left_index.has_value();
    ctx->saved_data["has_right_index"] = right_index.has_value();
    ctx->saved_data["fn"] = fn;
    ctx->save_for_backward({
        left, right,
        left_index.has_value() ? left_index.value() : left,     // dummy
        right_index.has_value() ? right_index.value() : right,  // dummy
    });
    return {out};
  }

  static variable_list backward(torch::autograd::AutogradContext* ctx,
                                variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();

    auto left = saved[0];
    auto right = saved[1];
    at::optional<at::Tensor> left_index = at::nullopt;
    if (ctx->saved_data["has_left_index"].toBool()) {
      left_index = saved[2];
    }
    at::optional<at::Tensor> right_index = at::nullopt;
    if (ctx->saved_data["has_right_index"].toBool()) {
      right_index = saved[3];
    }
    auto fn = ctx->saved_data["fn"].toStringRef();

    auto grad_left = at::Tensor();
    if (torch::autograd::any_variable_requires_grad({left})) {
      grad_left = grad_out;

      if (fn == "mul") {
        grad_left =
            sampled_op(grad_left, right, at::nullopt, right_index, "mul");
      } else if (fn == "div") {
        grad_left =
            sampled_op(grad_left, right, at::nullopt, right_index, "div");
      }

      if (left_index.has_value()) {
        grad_left = at::index_select_backward(grad_left, left.sizes(), 0,
                                              left_index.value());
      }
    }

    auto grad_right = at::Tensor();
    if (torch::autograd::any_variable_requires_grad({right})) {
      grad_right = grad_out;

      if (fn == "sub" && grad_out.size(0) <= right.size(0)) {
        grad_right = -grad_right;
      } else if (fn == "mul") {
        grad_right =
            sampled_op(grad_right, left, at::nullopt, left_index, "mul");
      } else if (fn == "div") {
        auto tmp = sampled_op(left, right, left_index, right_index, "div");
        tmp = sampled_op(tmp, right, at::nullopt, right_index, "div");
        grad_right = -grad_right * tmp;
      }

      if (right_index.has_value()) {
        grad_right = at::index_select_backward(grad_right, right.sizes(), 0,
                                               right_index.value());
      }

      if (fn == "sub" && grad_out.size(0) > right.size(0)) {
        grad_right = -grad_right;
      }
    }

    return {grad_left, grad_right, at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

at::Tensor sampled_op_autograd(const at::Tensor& left,
                               const at::Tensor& right,
                               const at::optional<at::Tensor> left_index,
                               const at::optional<at::Tensor> right_index,
                               const std::string fn) {
  return SampledOp::apply(left, right, left_index, right_index, fn)[0];
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, Autograd, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::sampled_op"),
         TORCH_FN(sampled_op_autograd));
}

}  // namespace ops
}  // namespace pyg
