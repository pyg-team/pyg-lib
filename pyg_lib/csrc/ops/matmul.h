#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace ops {

// Performs matrix multiplication across list of elements.
// TODO (matthias) Support `out` argument.
PYG_API std::vector<at::Tensor> grouped_matmul(
    const std::vector<at::Tensor>& input,
    const std::vector<at::Tensor>& other);

// Performs matrix multiplication according to segments.
// TODO (matthias) Support `out` argument.
PYG_API at::Tensor segment_matmul(const at::Tensor& input,
                                  const at::Tensor& ptr,
                                  const at::Tensor& other);

}  // namespace ops
}  // namespace pyg
