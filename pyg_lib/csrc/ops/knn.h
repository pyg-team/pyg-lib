#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace ops {

PYG_API at::Tensor knn(const at::Tensor& x,
                       const at::Tensor& y,
                       const std::optional<at::Tensor>& ptr_x,
                       const std::optional<at::Tensor>& ptr_y,
                       int64_t k,
                       bool cosine,
                       int64_t num_workers);

}  // namespace ops
}  // namespace pyg
