#include "softmax.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

// Performs softmax operations for each group.
PYG_API at::Tensor softmax_csr(const at::Tensor& src,
                               const at::Tensor& ptr,
                               const int64_t dim) {
  at::TensorArg src_arg{src, "src", 0};
  at::TensorArg ptr_arg{ptr, "ptr", 1};
  at::CheckedFrom c{"softmax_csr"};

  at::checkAllDefined(c, {src_arg, ptr_arg});
  at::checkContiguous(c, src_arg);
  at::checkContiguous(c, ptr_arg);

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::softmax_csr", "")
                       .typed<decltype(softmax_csr)>();
  return op.call(src, ptr, dim);
}

// Computes gradient for grouped softmax operation.
PYG_API at::Tensor softmax_csr_backward(const at::Tensor& out,
                                        const at::Tensor& out_grad,
                                        const at::Tensor& ptr,
                                        const int64_t dim) {
  at::TensorArg out_arg{out, "out", 0};
  at::TensorArg out_grad_arg{out_grad, "out_grad", 1};
  at::TensorArg ptr_arg{ptr, "ptr", 2};
  at::CheckedFrom c{"softmax_csr_backward"};

  at::checkAllDefined(c, {out_arg, out_grad_arg, ptr_arg});
  at::checkContiguous(c, out_arg);
  at::checkContiguous(c, out_grad_arg);
  at::checkContiguous(c, ptr_arg);

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::softmax_csr_backward", "")
                       .typed<decltype(softmax_csr_backward)>();
  return op.call(out, out_grad, ptr, dim);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(
      TORCH_SELECTIVE_SCHEMA("pyg::softmax_csr(Tensor src, Tensor ptr, "
                             "int dim=0) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::softmax_csr_backward(Tensor out, Tensor out_grad, "
      "Tensor ptr, int dim=0) -> Tensor"));
}

}  // namespace ops
}  // namespace pyg
