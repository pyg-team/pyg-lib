#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace segment {

// Performs matrix multiplication according to segments.
PYG_API at::Tensor matmul(const at::Tensor& input,
                          const at::Tensor& ptr,
                          const at::Tensor& other,
                          at::optional<at::Tensor&> out);

}  // namespace segment
}  // namespace pyg
