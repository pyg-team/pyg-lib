#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace pyg {
namespace sampler {

PYG_API at::Tensor random_walk(const at::Tensor& rowptr,
                               const at::Tensor& col,
                               const at::Tensor& seed,
                               int64_t walk_length,
                               double p,
                               double q);

}  // namespace sampler
}  // namespace pyg
