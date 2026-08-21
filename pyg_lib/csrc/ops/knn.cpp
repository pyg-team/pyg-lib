#include "knn.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

PYG_API at::Tensor knn(const at::Tensor& x,
                       const at::Tensor& y,
                       const std::optional<at::Tensor>& ptr_x,
                       const std::optional<at::Tensor>& ptr_y,
                       int64_t k,
                       bool cosine,
                       int64_t num_workers) {
  at::TensorArg x_arg{x, "x", 0};
  at::TensorArg y_arg{y, "y", 1};
  at::CheckedFrom c{"knn"};

  at::checkAllDefined(c, {x_arg, y_arg});
  at::checkDim(c, x_arg, 2);
  at::checkDim(c, y_arg, 2);

  TORCH_CHECK(x.size(1) == y.size(1), "x and y must have the same feature dim");
  TORCH_CHECK(k > 0, "k must be positive");

  auto x_c = x.contiguous();
  auto y_c = y.contiguous();

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::knn", "")
                       .typed<decltype(knn)>();
  return op.call(x_c, y_c, ptr_x, ptr_y, k, cosine, num_workers);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(
      TORCH_SELECTIVE_SCHEMA("pyg::knn(Tensor x, Tensor y, Tensor? ptr_x=None, "
                             "Tensor? ptr_y=None, int k=1, bool cosine=False, "
                             "int num_workers=1) -> Tensor"));
}

}  // namespace ops
}  // namespace pyg
