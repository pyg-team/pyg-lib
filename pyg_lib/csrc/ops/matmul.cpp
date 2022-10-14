#include "matmul.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

#include "pyg_lib/csrc/utils/check.h"

namespace pyg {
namespace ops {

// Performs matrix multiplication across list of elements.
std::vector<at::Tensor> grouped_matmul(const at::TensorList input,
                                       const at::TensorList other) {
  TORCH_CHECK(input.size() == other.size(),
              "Number of 'input' tensors must match number of 'other' tensors");

  std::vector<at::TensorArg> input_args;
  std::vector<at::TensorArg> other_args;
  pyg::utils::fill_tensor_args(input_args, input, "input", 0);
  pyg::utils::fill_tensor_args(other_args, other, "other", 1);
  at::CheckedFrom c{"grouped_matmul"};

  at::checkAllDefined(c, input_args);
  at::checkAllDefined(c, other_args);
  at::checkAllSameType(c, input_args);
  at::checkAllSameType(c, other_args);
  at::checkSameType(c, input_args[0], other_args[0]);
  for (size_t i = 0; i < input.size(); ++i) {
    at::checkDim(c, input_args[i], 2);
    at::checkDim(c, other_args[i], 2);
    at::checkSize(c, other_args[i], 0, input_args[i]->size(-1));
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::grouped_matmul", "")
                       .typed<decltype(grouped_matmul)>();
  return op.call(input, other);
}

// Performs matrix multiplication according to segments.
at::Tensor segment_matmul(const at::Tensor& input,
                          const at::Tensor& ptr,
                          const at::Tensor& other) {
  at::TensorArg input_arg{input, "input", 0};
  at::TensorArg ptr_arg{ptr, "ptr", 1};
  at::TensorArg other_arg{other, "other", 2};
  at::CheckedFrom c{"segment_matmul"};

  at::checkAllDefined(c, {input_arg, ptr_arg, other_arg});
  at::checkSameType(c, input_arg, other_arg);
  at::checkDim(c, input_arg, 2);
  at::checkDim(c, ptr_arg, 1);
  at::checkDim(c, other_arg, 3);
  at::checkSize(c, other_arg, 1, input_arg->size(-1));
  at::checkNumel(c, ptr_arg, other_arg->size(0) + 1);

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::segment_matmul", "")
                       .typed<decltype(segment_matmul)>();
  return op.call(input, ptr, other);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::grouped_matmul(Tensor[] input, Tensor[] other) -> Tensor[]"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::segment_matmul(Tensor input, Tensor ptr, Tensor other) -> Tensor"));
}

}  // namespace ops
}  // namespace pyg
