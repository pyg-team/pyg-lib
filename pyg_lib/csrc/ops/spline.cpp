#include "spline.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

PYG_API std::tuple<at::Tensor, at::Tensor> spline_basis(
    const at::Tensor& pseudo,
    const at::Tensor& kernel_size,
    const at::Tensor& is_open_spline,
    int64_t degree) {
  at::TensorArg pseudo_arg{pseudo, "pseudo", 0};
  at::TensorArg kernel_size_arg{kernel_size, "kernel_size", 1};
  at::TensorArg is_open_spline_arg{is_open_spline, "is_open_spline", 2};
  at::CheckedFrom c{"spline_basis"};

  at::checkAllDefined(c, {pseudo_arg, kernel_size_arg, is_open_spline_arg});
  at::checkDim(c, pseudo_arg, 2);
  at::checkDim(c, kernel_size_arg, 1);
  at::checkDim(c, is_open_spline_arg, 1);

  TORCH_CHECK(pseudo.size(1) == kernel_size.numel(),
              "pseudo.size(1) must equal kernel_size.numel()");
  TORCH_CHECK(pseudo.size(1) == is_open_spline.numel(),
              "pseudo.size(1) must equal is_open_spline.numel()");

  auto pseudo_c = pseudo.contiguous();

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::spline_basis", "")
                       .typed<decltype(spline_basis)>();
  return op.call(pseudo_c, kernel_size, is_open_spline, degree);
}

PYG_API at::Tensor spline_basis_backward(const at::Tensor& grad_basis,
                                         const at::Tensor& pseudo,
                                         const at::Tensor& kernel_size,
                                         const at::Tensor& is_open_spline,
                                         int64_t degree) {
  at::TensorArg grad_basis_arg{grad_basis, "grad_basis", 0};
  at::TensorArg pseudo_arg{pseudo, "pseudo", 1};
  at::TensorArg kernel_size_arg{kernel_size, "kernel_size", 2};
  at::TensorArg is_open_spline_arg{is_open_spline, "is_open_spline", 3};
  at::CheckedFrom c{"spline_basis_backward"};

  at::checkAllDefined(
      c, {grad_basis_arg, pseudo_arg, kernel_size_arg, is_open_spline_arg});
  at::checkDim(c, grad_basis_arg, 2);
  at::checkDim(c, pseudo_arg, 2);
  at::checkDim(c, kernel_size_arg, 1);
  at::checkDim(c, is_open_spline_arg, 1);

  TORCH_CHECK(grad_basis.size(0) == pseudo.size(0),
              "grad_basis.size(0) must equal pseudo.size(0)");
  TORCH_CHECK(pseudo.size(1) == kernel_size.numel(),
              "pseudo.size(1) must equal kernel_size.numel()");
  TORCH_CHECK(pseudo.size(1) == is_open_spline.numel(),
              "pseudo.size(1) must equal is_open_spline.numel()");

  auto grad_basis_c = grad_basis.contiguous();
  auto pseudo_c = pseudo.contiguous();

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::spline_basis_backward", "")
                       .typed<decltype(spline_basis_backward)>();
  return op.call(grad_basis_c, pseudo_c, kernel_size, is_open_spline, degree);
}

PYG_API at::Tensor spline_weighting(const at::Tensor& x,
                                    const at::Tensor& weight,
                                    const at::Tensor& basis,
                                    const at::Tensor& weight_index) {
  at::TensorArg x_arg{x, "x", 0};
  at::TensorArg weight_arg{weight, "weight", 1};
  at::TensorArg basis_arg{basis, "basis", 2};
  at::TensorArg weight_index_arg{weight_index, "weight_index", 3};
  at::CheckedFrom c{"spline_weighting"};

  at::checkAllDefined(c, {x_arg, weight_arg, basis_arg, weight_index_arg});
  at::checkDim(c, x_arg, 2);
  at::checkDim(c, weight_arg, 3);
  at::checkDim(c, basis_arg, 2);
  at::checkDim(c, weight_index_arg, 2);

  TORCH_CHECK(x.size(1) == weight.size(1),
              "x.size(1) must equal weight.size(1)");
  TORCH_CHECK(x.size(0) == basis.size(0), "x.size(0) must equal basis.size(0)");
  TORCH_CHECK(x.size(0) == weight_index.size(0),
              "x.size(0) must equal weight_index.size(0)");
  TORCH_CHECK(basis.size(1) == weight_index.size(1),
              "basis.size(1) must equal weight_index.size(1)");

  auto x_c = x.contiguous();
  auto weight_c = weight.contiguous();
  auto basis_c = basis.contiguous();
  auto weight_index_c = weight_index.contiguous();

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::spline_weighting", "")
                       .typed<decltype(spline_weighting)>();
  return op.call(x_c, weight_c, basis_c, weight_index_c);
}

PYG_API at::Tensor spline_weighting_backward_x(const at::Tensor& grad_out,
                                               const at::Tensor& weight,
                                               const at::Tensor& basis,
                                               const at::Tensor& weight_index) {
  at::TensorArg grad_out_arg{grad_out, "grad_out", 0};
  at::TensorArg weight_arg{weight, "weight", 1};
  at::TensorArg basis_arg{basis, "basis", 2};
  at::TensorArg weight_index_arg{weight_index, "weight_index", 3};
  at::CheckedFrom c{"spline_weighting_backward_x"};

  at::checkAllDefined(c,
                      {grad_out_arg, weight_arg, basis_arg, weight_index_arg});

  auto grad_out_c = grad_out.contiguous();
  auto weight_c = weight.contiguous();
  auto basis_c = basis.contiguous();
  auto weight_index_c = weight_index.contiguous();

  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("pyg::spline_weighting_backward_x", "")
          .typed<decltype(spline_weighting_backward_x)>();
  return op.call(grad_out_c, weight_c, basis_c, weight_index_c);
}

PYG_API at::Tensor spline_weighting_backward_weight(
    const at::Tensor& grad_out,
    const at::Tensor& x,
    const at::Tensor& basis,
    const at::Tensor& weight_index,
    int64_t kernel_size) {
  at::TensorArg grad_out_arg{grad_out, "grad_out", 0};
  at::TensorArg x_arg{x, "x", 1};
  at::TensorArg basis_arg{basis, "basis", 2};
  at::TensorArg weight_index_arg{weight_index, "weight_index", 3};
  at::CheckedFrom c{"spline_weighting_backward_weight"};

  at::checkAllDefined(c, {grad_out_arg, x_arg, basis_arg, weight_index_arg});

  auto grad_out_c = grad_out.contiguous();
  auto x_c = x.contiguous();
  auto basis_c = basis.contiguous();
  auto weight_index_c = weight_index.contiguous();

  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("pyg::spline_weighting_backward_weight", "")
          .typed<decltype(spline_weighting_backward_weight)>();
  return op.call(grad_out_c, x_c, basis_c, weight_index_c, kernel_size);
}

PYG_API at::Tensor spline_weighting_backward_basis(
    const at::Tensor& grad_out,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& weight_index) {
  at::TensorArg grad_out_arg{grad_out, "grad_out", 0};
  at::TensorArg x_arg{x, "x", 1};
  at::TensorArg weight_arg{weight, "weight", 2};
  at::TensorArg weight_index_arg{weight_index, "weight_index", 3};
  at::CheckedFrom c{"spline_weighting_backward_basis"};

  at::checkAllDefined(c, {grad_out_arg, x_arg, weight_arg, weight_index_arg});

  auto grad_out_c = grad_out.contiguous();
  auto x_c = x.contiguous();
  auto weight_c = weight.contiguous();
  auto weight_index_c = weight_index.contiguous();

  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("pyg::spline_weighting_backward_basis", "")
          .typed<decltype(spline_weighting_backward_basis)>();
  return op.call(grad_out_c, x_c, weight_c, weight_index_c);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::spline_basis(Tensor pseudo, Tensor kernel_size, "
      "Tensor is_open_spline, int degree=1) -> (Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::spline_basis_backward(Tensor grad_basis, Tensor pseudo, "
      "Tensor kernel_size, Tensor is_open_spline, int degree=1) -> Tensor"));
  m.def(
      TORCH_SELECTIVE_SCHEMA("pyg::spline_weighting(Tensor x, Tensor weight, "
                             "Tensor basis, Tensor weight_index) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::spline_weighting_backward_x(Tensor grad_out, Tensor weight, "
      "Tensor basis, Tensor weight_index) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::spline_weighting_backward_weight(Tensor grad_out, Tensor x, "
      "Tensor basis, Tensor weight_index, int kernel_size) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::spline_weighting_backward_basis(Tensor grad_out, Tensor x, "
      "Tensor weight, Tensor weight_index) -> Tensor"));
}

}  // namespace ops
}  // namespace pyg
