#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace ops {

PYG_API at::Tensor radius(const at::Tensor& x,
                          const at::Tensor& y,
                          const std::optional<at::Tensor>& ptr_x,
                          const std::optional<at::Tensor>& ptr_y,
                          double r,
                          int64_t max_num_neighbors,
                          int64_t num_workers,
                          bool ignore_same_index);

}  // namespace ops
}  // namespace pyg
