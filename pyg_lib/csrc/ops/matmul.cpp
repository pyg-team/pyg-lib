#include "matmul.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/script.h>

namespace pyg {
namespace ops {

namespace {

std::vector<at::Tensor> _grouped_matmul(const std::vector<at::Tensor>& input,
                                        const std::vector<at::Tensor>& other) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::grouped_matmul", "")
                       .typed<decltype(grouped_matmul)>();
  // TODO (matthias) Add TensorArg definitions.
  // TODO (matthias) Add autograd support.
  // TODO (matthias) Add dispatcher support.
  // TODO (rishi) Add get GroupedMatmul backward working
  return op.call(input, other);
}

at::Tensor _segment_matmul(const at::Tensor& input,
                           const at::Tensor& ptr,
                           const at::Tensor& other) {
  // TODO (matthias) Add TensorArg definitions.
  // TODO (matthias) Add autograd support.
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::segment_matmul", "")
                       .typed<decltype(segment_matmul)>();
  return op.call(input, ptr, other);
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

// Performs matrix multiplication according to segments.
class SegmentMatmul : public torch::autograd::Function<SegmentMatmul> {
 public:
  static variable_list forward(AutogradContext* ctx,
                               Variable input,
                               const at::Tensor& ptr,
                               Variable other) {
    Variable out = _segment_matmul(input, ptr, other);
    ctx->save_for_backward({input, ptr, other});
    return {out};
  }

  static variable_list backward(AutogradContext* ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto input = saved[0], ptr = saved[1], other = saved[2];

    auto input_grad = Variable(), other_grad = Variable();

    if (torch::autograd::any_variable_requires_grad({input})) {
      // TODO (matthias) get rid of unnecessary `contiguous` here.
      auto input_t = input.transpose(-2, -1).contiguous();
      input_grad = _segment_matmul(input_t, ptr, grad_out);
    }

    if (torch::autograd::any_variable_requires_grad({other})) {
      // TODO (matthias) get rid of unnecessary `contiguous` here.
      auto other_t = other.transpose(-2, -1).contiguous();
      other_grad = _segment_matmul(grad_out, ptr, other_t);
    }

    return {input_grad, other_grad};
  }
};

}  // namespace

// Performs matrix multiplication across list of elements.
std::vector<at::Tensor> grouped_matmul(const std::vector<at::Tensor>& input,
                                       const std::vector<at::Tensor>& other) {
  return _grouped_matmul(input, other);
}

// Performs matrix multiplication according to segments.
at::Tensor segment_matmul(const at::Tensor& input,
                          const at::Tensor& ptr,
                          const at::Tensor& other) {
  return SegmentMatmul::apply(input, ptr, other)[0];
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::grouped_matmul(Tensor[] input, Tensor[] other) -> Tensor[]"));
  m.def(
      TORCH_SELECTIVE_SCHEMA("pyg::segment_matmul(Tensor input, Tensor ptr, "
                             "Tensor other) -> Tensor"));
}
}  // namespace ops
}  // namespace pyg
