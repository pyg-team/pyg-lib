#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace ops {

PYG_API at::Tensor grid_cluster(const at::Tensor& pos,
                                const at::Tensor& size,
                                const std::optional<at::Tensor>& start,
                                const std::optional<at::Tensor>& end);

}  // namespace ops
}  // namespace pyg
