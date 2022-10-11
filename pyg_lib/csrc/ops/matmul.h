#pragma once

#include <ATen/ATen.h>
#include <torch/script.h>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace ops {

using torch::autograd::Variable;
using torch::autograd::variable_list;

// Performs matrix multiplication across list of elements.
// TODO (matthias) Support `out` argument.
PYG_API std::vector<at::Tensor> grouped_matmul(const variable_list input,
                                               const variable_list other);

// Performs matrix multiplication according to segments.
// TODO (matthias) Support `out` argument.
PYG_API at::Tensor segment_matmul(const Variable input,
                                  const at::Tensor& ptr,
                                  const Variable other);

}  // namespace ops
}  // namespace pyg
