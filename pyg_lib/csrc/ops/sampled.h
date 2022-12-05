#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace ops {

// Performs the operation `op` at sampled left and right indices.
PYG_API at::Tensor sampled_op(const at::Tensor& left,
                              const at::Tensor& right,
                              const at::optional<at::Tensor> left_index,
                              const at::optional<at::Tensor> right_index,
                              const std::string fn);

}  // namespace ops
}  // namespace pyg
