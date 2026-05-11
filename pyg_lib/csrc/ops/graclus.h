#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace ops {

PYG_API at::Tensor graclus_cluster(const at::Tensor& rowptr,
                                   const at::Tensor& col,
                                   const std::optional<at::Tensor>& weight);

}  // namespace ops
}  // namespace pyg
