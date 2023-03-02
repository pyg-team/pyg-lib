#include "sampled.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

// Performs the operation `op` at sampled left and right indices.
PYG_API at::Tensor sampled_op(const at::Tensor& left,
                              const at::Tensor& right,
                              const at::optional<at::Tensor> left_index,
                              const at::optional<at::Tensor> right_index,
                              const std::string fn) {
  at::TensorArg left_arg{left, "left", 0};
  at::TensorArg right_arg{right, "right", 1};
  at::CheckedFrom c{"sampled_op"};

  at::checkAllDefined(c, {left_arg, right_arg});
  at::checkSameType(c, left_arg, right_arg);
  at::checkContiguous(c, left_arg);
  at::checkContiguous(c, right_arg);
  at::checkDim(c, left_arg, 2);
  at::checkDim(c, right_arg, 2);
  at::checkSize(c, left_arg, 1, right_arg->size(1));

  if (left_index.has_value()) {
    at::TensorArg left_index_arg{left_index.value(), "left_index", 2};
    at::checkContiguous(c, left_index_arg);
    at::checkDim(c, left_index_arg, 1);
  }

  if (right_index.has_value()) {
    at::TensorArg right_index_arg{right_index.value(), "right_index", 3};
    at::checkContiguous(c, right_index_arg);
    at::checkDim(c, right_index_arg, 1);
  }

  if (left_index.has_value() && right_index.has_value()) {
    at::TensorArg left_index_arg{left_index.value(), "left_index", 2};
    at::TensorArg right_index_arg{right_index.value(), "right_index", 3};
    at::checkSameType(c, left_index_arg, right_index_arg);
    at::checkSize(c, left_index_arg, 0, right_index_arg->size(0));
  }

  if (!left_index.has_value() && !right_index.has_value()) {
    at::checkSize(c, left_arg, 0, right_arg->size(0));
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::sampled_op", "")
                       .typed<decltype(sampled_op)>();
  return op.call(left, right, left_index, right_index, fn);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::sampled_op(Tensor left, Tensor right, Tensor? left_index, Tensor? "
      "right_index, str op) -> Tensor"));
}

}  // namespace ops
}  // namespace pyg
