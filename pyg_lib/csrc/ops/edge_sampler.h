#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace ops {

PYG_API at::Tensor edge_sample(const at::Tensor& start,
                               const at::Tensor& rowptr,
                               int64_t count,
                               double factor);

}  // namespace ops
}  // namespace pyg
