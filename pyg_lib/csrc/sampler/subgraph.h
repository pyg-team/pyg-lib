#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace sampler {

PYG_API std::tuple<at::Tensor, at::Tensor> subgraph(const at::Tensor& rowptr,
                                                    const at::Tensor& col,
                                                    const at::Tensor& nodes);

}  // namespace sampler
}  // namespace pyg
