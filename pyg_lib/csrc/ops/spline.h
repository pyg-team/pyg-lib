#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace ops {

PYG_API std::tuple<at::Tensor, at::Tensor> spline_basis(
    const at::Tensor& pseudo,
    const at::Tensor& kernel_size,
    const at::Tensor& is_open_spline,
    int64_t degree);

PYG_API at::Tensor spline_basis_backward(const at::Tensor& grad_basis,
                                         const at::Tensor& pseudo,
                                         const at::Tensor& kernel_size,
                                         const at::Tensor& is_open_spline,
                                         int64_t degree);

PYG_API at::Tensor spline_weighting(const at::Tensor& x,
                                    const at::Tensor& weight,
                                    const at::Tensor& basis,
                                    const at::Tensor& weight_index);

PYG_API at::Tensor spline_weighting_backward_x(const at::Tensor& grad_out,
                                               const at::Tensor& weight,
                                               const at::Tensor& basis,
                                               const at::Tensor& weight_index);

PYG_API at::Tensor spline_weighting_backward_weight(
    const at::Tensor& grad_out,
    const at::Tensor& x,
    const at::Tensor& basis,
    const at::Tensor& weight_index,
    int64_t kernel_size);

PYG_API at::Tensor spline_weighting_backward_basis(
    const at::Tensor& grad_out,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& weight_index);

}  // namespace ops
}  // namespace pyg
