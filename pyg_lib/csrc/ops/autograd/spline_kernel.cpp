#include "../spline.h"

#include <torch/autograd.h>

namespace pyg {
namespace ops {

namespace {

using torch::autograd::variable_list;

class SplineBasis : public torch::autograd::Function<SplineBasis> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& pseudo,
                               const at::Tensor& kernel_size,
                               const at::Tensor& is_open_spline,
                               const int64_t degree) {
    at::AutoDispatchBelowADInplaceOrView g;

    auto result = spline_basis(pseudo, kernel_size, is_open_spline, degree);
    auto basis = std::get<0>(result);
    auto weight_index = std::get<1>(result);

    ctx->saved_data["degree"] = degree;
    ctx->save_for_backward({pseudo, kernel_size, is_open_spline});
    ctx->mark_non_differentiable({weight_index});

    return {basis, weight_index};
  }

  static variable_list backward(torch::autograd::AutogradContext* ctx,
                                variable_list grad_outs) {
    const auto grad_basis = grad_outs[0];
    const auto saved = ctx->get_saved_variables();
    const auto pseudo = saved[0];
    const auto kernel_size = saved[1];
    const auto is_open_spline = saved[2];
    const auto degree = ctx->saved_data["degree"].toInt();

    auto grad_pseudo = at::Tensor();
    if (torch::autograd::any_variable_requires_grad({pseudo})) {
      grad_pseudo = spline_basis_backward(grad_basis, pseudo, kernel_size,
                                          is_open_spline, degree);
    }

    return {grad_pseudo, at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

class SplineWeighting : public torch::autograd::Function<SplineWeighting> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& x,
                               const at::Tensor& weight,
                               const at::Tensor& basis,
                               const at::Tensor& weight_index) {
    at::AutoDispatchBelowADInplaceOrView g;

    auto out = spline_weighting(x, weight, basis, weight_index);

    ctx->save_for_backward({x, weight, basis, weight_index});

    return {out};
  }

  static variable_list backward(torch::autograd::AutogradContext* ctx,
                                variable_list grad_outs) {
    const auto grad_out = grad_outs[0];
    const auto saved = ctx->get_saved_variables();
    const auto x = saved[0];
    const auto weight = saved[1];
    const auto basis = saved[2];
    const auto weight_index = saved[3];

    auto grad_x = at::Tensor();
    if (torch::autograd::any_variable_requires_grad({x})) {
      grad_x =
          spline_weighting_backward_x(grad_out, weight, basis, weight_index);
    }

    auto grad_weight = at::Tensor();
    if (torch::autograd::any_variable_requires_grad({weight})) {
      grad_weight = spline_weighting_backward_weight(
          grad_out, x, basis, weight_index, weight.size(0));
    }

    auto grad_basis = at::Tensor();
    if (torch::autograd::any_variable_requires_grad({basis})) {
      grad_basis =
          spline_weighting_backward_basis(grad_out, x, weight, weight_index);
    }

    return {grad_x, grad_weight, grad_basis, at::Tensor()};
  }
};

std::tuple<at::Tensor, at::Tensor> spline_basis_autograd(
    const at::Tensor& pseudo,
    const at::Tensor& kernel_size,
    const at::Tensor& is_open_spline,
    int64_t degree) {
  auto result = SplineBasis::apply(pseudo, kernel_size, is_open_spline, degree);
  return std::make_tuple(result[0], result[1]);
}

at::Tensor spline_weighting_autograd(const at::Tensor& x,
                                     const at::Tensor& weight,
                                     const at::Tensor& basis,
                                     const at::Tensor& weight_index) {
  return SplineWeighting::apply(x, weight, basis, weight_index)[0];
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, Autograd, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::spline_basis"),
         TORCH_FN(spline_basis_autograd));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spline_weighting"),
         TORCH_FN(spline_weighting_autograd));
}

}  // namespace ops
}  // namespace pyg
