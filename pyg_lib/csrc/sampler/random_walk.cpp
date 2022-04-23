#include "random_walk.h"

#include <torch/library.h>

namespace pyg {
namespace sampler {

at::Tensor random_walk(const at::Tensor& crow,
                       const at::Tensor& col,
                       const at::Tensor& seed,
                       int64_t walk_length,
                       double p,
                       double q) {
  return seed;
}

/* TORCH_LIBRARY_FRAGMENT(pyg, m) { */
/*   m.def(TORCH_SELECTIVE_SCHEMA( */
/*       "pyg:random_walk(Tensor crow, Tensor col, Tensor seed, int walk_length,
 * " */
/*       "float p, float q) -> Tensor")); */
/* } */

}  // namespace sampler
}  // namespace pyg
