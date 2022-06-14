#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace segment {

// Performs matrix multiplication across list of elements.
// TODO (matthias) Import `out` argument.
PYG_API std::vector<at::Tensor> grouped_matmul(
    const std::vector<at::Tensor>& input,
    const std::vector<at::Tensor>& other);

// TODO (matthias) Import `out` argument.
PYG_API at::Tensor segment_matmul(const at::Tensor& input,
                                  const at::Tensor& ptr,
                                  const at::Tensor& other);

}  // namespace segment
}  // namespace pyg
