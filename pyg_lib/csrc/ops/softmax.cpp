#include "softmax.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

// Performs softmax operations for each group.
PYG_API at::Tensor softmax_forward(const at::Tensor& src,
                                   const at::optional<at::Tensor> index,
                                   const at::optional<at::Tensor> ptr,
                                   const at::optional<int64_t> num_nodes,
                                   const int64_t dim) {
  at::TensorArg src_arg{src, "src", 0};
  at::CheckedFrom c{"softmax_forward"};

  at::checkAllDefined(c, {src_arg});
  at::checkContiguous(c, src_arg);

  if (index.has_value()) {
    at::TensorArg index_arg{index.value(), "index", 1};
    at::checkContiguous(c, index_arg);
  }

  if (ptr.has_value()) {
    at::TensorArg ptr_arg{ptr.value(), "ptr", 2};
    at::checkContiguous(c, ptr_arg);
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::softmax_forward", "")
                       .typed<decltype(softmax_forward)>();
  return op.call(src, index, ptr, num_nodes, dim);
}

// Computes gradient for grouped softmax operation.
PYG_API at::Tensor softmax_backward(const at::Tensor& out,
                                    const at::Tensor& out_grad,
                                    const at::optional<at::Tensor> index,
                                    const at::optional<at::Tensor> ptr,
                                    const at::optional<int64_t> num_nodes,
                                    const int64_t dim) {
  at::TensorArg out_arg{out, "out", 0};
  at::TensorArg out_grad_arg{out_grad, "out_grad", 1};
  at::CheckedFrom c{"softmax_backward"};

  at::checkAllDefined(c, {out_arg, out_grad_arg});
  at::checkContiguous(c, out_arg);
  at::checkContiguous(c, out_grad_arg);

  if (index.has_value()) {
    at::TensorArg index_arg{index.value(), "index", 2};
    at::checkContiguous(c, index_arg);
  }

  if (ptr.has_value()) {
    at::TensorArg ptr_arg{ptr.value(), "ptr", 3};
    at::checkContiguous(c, ptr_arg);
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::softmax_backward", "")
                       .typed<decltype(softmax_backward)>();
  return op.call(out, out_grad, index, ptr, num_nodes, dim);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::softmax_forward(Tensor src, Tensor? index, Tensor? ptr, "
      "int? num_nodes, int dim=0) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::softmax_backward(Tensor out, Tensor out_grad, Tensor? index, "
      "Tensor? ptr, int? num_nodes, int dim=0) -> Tensor"));
}

}  // namespace ops
}  // namespace pyg
