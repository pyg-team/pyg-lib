#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace ops {

PYG_API at::Tensor fps(const at::Tensor& src,
                       const at::Tensor& ptr,
                       double ratio,
                       bool random_start);

}  // namespace ops
}  // namespace pyg
