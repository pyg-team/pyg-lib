#include "matmul.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace segment {

// Performs matrix multiplication according to segments.
at::Tensor matmul(const at::Tensor& input,
                  const at::Tensor& ptr,
                  const at::Tensor& other,
                  const at::Tensor& out) {
  at::TensorArg input_t{input, "input", 2};
  at::TensorArg ptr_t{ptr, "ptr", 1};
  at::TensorArg other_t{other, "other", 3};
  at::TensorArg out_t{out, "out", 2};

  at::CheckedFrom c = "segment_matmul";
  at::checkAllDefined(c, {input_t, ptr_t, other_t, out_t});
  at::checkAllSameType(c, {input_t, other_t, out_t});

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::segment_matmul", "")
                       .typed<decltype(matmul)>();
  return op.call(input, ptr, other, out);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(
      TORCH_SELECTIVE_SCHEMA("pyg::segment_matmul(Tensor input, Tensor ptr, "
                             "Tensor other, Tensor out) -> Tensor"));
}

}  // namespace segment
}  // namespace pyg
