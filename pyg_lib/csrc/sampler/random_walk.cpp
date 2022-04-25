#include "random_walk.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace sampler {

at::Tensor random_walk(const at::Tensor& rowptr,
                       const at::Tensor& col,
                       const at::Tensor& seed,
                       int64_t walk_length,
                       double p,
                       double q) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::random_walk", "")
                       .typed<decltype(random_walk)>();
  return op.call(rowptr, col, seed, walk_length, p, q);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::random_walk(Tensor rowptr, Tensor col, Tensor seed, int "
      "walk_length, float p, float q) -> Tensor"));
}

}  // namespace sampler
}  // namespace pyg
