#include "fps.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

PYG_API at::Tensor fps(const at::Tensor& src,
                       const at::Tensor& ptr,
                       double ratio,
                       bool random_start) {
  at::TensorArg src_arg{src, "src", 0};
  at::TensorArg ptr_arg{ptr, "ptr", 1};
  at::CheckedFrom c{"fps"};

  at::checkAllDefined(c, {src_arg, ptr_arg});
  at::checkDim(c, ptr_arg, 1);

  TORCH_CHECK(ratio > 0.0 && ratio <= 1.0, "ratio must be in the range (0, 1]");

  auto src_c = src.view({src.size(0), -1}).contiguous();
  auto ptr_c = ptr.contiguous();

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::fps", "")
                       .typed<decltype(fps)>();
  return op.call(src_c, ptr_c, ratio, random_start);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::fps(Tensor src, Tensor ptr, float ratio=0.5, "
      "bool random_start=True) -> Tensor"));
}

}  // namespace ops
}  // namespace pyg
