#include "nearest.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

PYG_API at::Tensor nearest(const at::Tensor& x,
                           const at::Tensor& y,
                           const std::optional<at::Tensor>& ptr_x,
                           const std::optional<at::Tensor>& ptr_y) {
  at::TensorArg x_arg{x, "x", 0};
  at::TensorArg y_arg{y, "y", 1};
  at::CheckedFrom c{"nearest"};

  at::checkAllDefined(c, {x_arg, y_arg});

  auto x_c = x.view({x.size(0), -1}).contiguous();
  auto y_c = y.view({y.size(0), -1}).contiguous();

  TORCH_CHECK(x_c.size(1) == y_c.size(1),
              "x and y must have the same feature dim");

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::nearest", "")
                       .typed<decltype(nearest)>();
  return op.call(x_c, y_c, ptr_x, ptr_y);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::nearest(Tensor x, Tensor y, Tensor? ptr_x=None, "
      "Tensor? ptr_y=None) -> Tensor"));
}

}  // namespace ops
}  // namespace pyg
