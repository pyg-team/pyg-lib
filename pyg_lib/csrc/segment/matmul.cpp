#include "matmul.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace segment {

// Performs matrix multiplication across list of elements.
std::vector<at::Tensor> grouped_matmul(const std::vector<at::Tensor>& input,
                                       const std::vector<at::Tensor>& other) {
  // TODO (matthias) Add TensorArg definitions.
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::grouped_matmul", "")
                       .typed<decltype(grouped_matmul)>();
  return op.call(input, other);
}

// Performs matrix multiplication according to segments.
at::Tensor segment_matmul(const at::Tensor& input,
                          const at::Tensor& ptr,
                          const at::Tensor& other) {
  // TODO (matthias) Add TensorArg definitions.
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::segment_matmul", "")
                       .typed<decltype(segment_matmul)>();
  return op.call(input, ptr, other);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::grouped_matmul(Tensor[] input, Tensor[] other) -> Tensor[]"));
  m.def(
      TORCH_SELECTIVE_SCHEMA("pyg::segment_matmul(Tensor input, Tensor ptr, "
                             "Tensor other) -> Tensor"));
}

}  // namespace segment
}  // namespace pyg
