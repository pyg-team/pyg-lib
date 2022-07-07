#include "matmul.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

// Performs matrix multiplication across list of elements.
class GroupedMatmul : public torch::autograd::Function<GroupedMatmul> {
 public:
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::grouped_matmul", "")
                       .typed<decltype(grouped_matmul)>();
  static variable_list forward(AutogradContext* ctx,
                               std::vector<Variable> input,
                               std::vector<Variable> other) {
    auto out =
        op.call(input, other) ctx->save_for_backward({out, input, other});
    return out;
  }

  static variable_list backward(AutogradContext* ctx, variable_list grad_outs) {
    auto saved = ctx->get_saved_variables();
    auto out = saved[0];
    auto input = saved[1]; 
    for (size_t i = 0; i < input.size(); ++i)
      input[i] = input[i].transpose(-2, -1);
    auto other = saved[2];
    for (size_t i = 0; i < input.size(); ++i)
      other[i] = other[i].transpose(-2, -1);
    auto input_grad = op.call(input, grad_outs);
    auto other_grad = op.call(grad_outs, other)
    return {input_grad, other_grad};
  }
};

std::vector<at::Tensor> grouped_matmul(const std::vector<at::Tensor>& input,
                                       const std::vector<at::Tensor>& other) {
  return GroupedMatmul::apply(input, other);
}

// Performs matrix multiplication according to segments.
class SegmentMatmul : public torch::autograd::Function<SegmentMatmul> {
 public:
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::segment_matmul", "")
                       .typed<decltype(segment_matmul)>();
  static Variable forward(AutogradContext* ctx,
                           Variable input,
                           const at::Tensor& ptr,
                           Variable other) {
    auto out = op.call(input, ptr, other);
    ctx->save_for_backward({out, input, ptr, other});
    return out;
  }

  static Variable backward(AutogradContext* ctx, Variable grad_out) {
    auto saved = ctx->get_saved_variables();
    auto out = saved[0];
    auto input = saved[1].transpose(-2, -1);
    auto ptr = saved[2];
    auto other = saved[3].transpose(-2, -1);
    auto input_grad = op.call(input, ptr, grad_outs);
    auto other_grad = op.call(grad_outs, ptr, other);
    return {input_grad, other_grad};
  }
};

std::vector<at::Tensor> segment_matmul(const at::Tensor& input,
                                       const at::Tensor& ptr,
                                       const at::Tensor& other) {
  return SegmentMatmul::apply(input, ptr, other);
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
