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
  at::TensorArg rowptr_t{rowptr, "rowtpr", 1};
  at::TensorArg col_t{col, "col", 1};
  at::TensorArg seed_t{seed, "seed", 1};

  at::CheckedFrom c = "random_walk";
  at::checkAllDefined(c, {rowptr_t, col_t, seed_t});
  at::checkAllSameType(c, {rowptr_t, col_t, seed_t});

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
