#pragma once

#include <torch/torch.h>
#include "../macros.h"

namespace pyg {
namespace sampler {

PYG_API torch::Tensor random_walk(const torch::Tensor& rowptr,
                                  const torch::Tensor& col,
                                  const torch::Tensor& seed,
                                  int64_t walk_length,
                                  double p = 1.0,
                                  double q = 1.0);

}  // namespace sampler
}  // namespace pyg
