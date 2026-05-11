#include "radius.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

PYG_API at::Tensor radius(const at::Tensor& x,
                          const at::Tensor& y,
                          const std::optional<at::Tensor>& ptr_x,
                          const std::optional<at::Tensor>& ptr_y,
                          double r,
                          int64_t max_num_neighbors,
                          int64_t num_workers,
                          bool ignore_same_index) {
  at::TensorArg x_arg{x, "x", 0};
  at::TensorArg y_arg{y, "y", 1};
  at::CheckedFrom c{"radius"};

  at::checkAllDefined(c, {x_arg, y_arg});
  at::checkDim(c, x_arg, 2);
  at::checkDim(c, y_arg, 2);
  TORCH_CHECK(x.size(1) == y.size(1), "x and y must have the same feature dim");
  TORCH_CHECK(r > 0, "r must be positive");

  auto x_c = x.contiguous();
  auto y_c = y.contiguous();

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::radius", "")
                       .typed<decltype(radius)>();
  return op.call(x_c, y_c, ptr_x, ptr_y, r, max_num_neighbors, num_workers,
                 ignore_same_index);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::radius(Tensor x, Tensor y, Tensor? ptr_x=None, "
      "Tensor? ptr_y=None, float r=1.0, int max_num_neighbors=32, "
      "int num_workers=1, bool ignore_same_index=False) -> Tensor"));
}

}  // namespace ops
}  // namespace pyg
