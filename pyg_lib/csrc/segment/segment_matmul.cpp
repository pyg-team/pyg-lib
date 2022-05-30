#include "segment_matmul.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace segment {

// Performs matrix multiplication according to segments.
PYG_API at::Tensor matmul(const at::Tensor& input,
                          const at::Tensor& ptr,
                          const at::Tensor& other,
                          at::optional<at::Tensor&> out) {
  at::TensorArg input_t{input, "input", 2};
  at::TensorArg ptr_t{ptr, "ptr", 1};
  at::TensorArg other_t{other, "other", 3};

  at::CheckedFrom c = "segment_matmul";
  at::checkAllDefined(c, {input_t, ptr_t, other_t});
  at::checkAllSameType(c, {input_t, ptr_t, other_t});

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::segment_matmul", "")
                       .typed<decltype(segment_matmul)>();
  return op.call(input, ptr, other, out);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(
      TORCH_SELECTIVE_SCHEMA("pyg::segment_matmul(Tensor input, Tensor ptr, "
                             "Tensor other, Tensor? out) -> Tensor"));
}

}  // namespace segment
}  // namespace pyg
