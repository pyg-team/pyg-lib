#include "matmul.h"
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/script.h>

namespace pyg {
namespace ops {

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

static auto group_op = c10::Dispatcher::singleton()
                           .findSchemaOrThrow("pyg::grouped_matmul", "")
                           .typed<decltype(grouped_matmul)>();

static auto segment_op = c10::Dispatcher::singleton()
                             .findSchemaOrThrow("pyg::segment_matmul", "")
                             .typed<decltype(segment_matmul)>();

// Performs matrix multiplication across list of elements.
class GroupedMatmul : public torch::autograd::Function<GroupedMatmul> {
  // TODO (matthias) Add TensorArg definitions.
 public:
  static variable_list forward(AutogradContext* ctx,
                               std::vector<Variable> input,
                               std::vector<Variable> other) {
    auto out = group_op.call(input, other);
    // ctx->save_for_backward({input, other});
    return out;
  }

  // TODO (rishi) Add GroupedMatmul backward
  // static variable_list backward(AutogradContext* ctx, variable_list
  // grad_outs) {
  //   auto saved = ctx->get_saved_variables();
  //   variable_list input = saved[0];
  //   variable_list other = saved[1];
  //   for (size_t i = 0; i < input.size(); ++i)
  //     other[i] = other[i].transpose(-2, -1).contiguous();
  //   auto other_grad = group_op.call(grad_outs, other);
  //   if (torch::autograd::any_variable_requires_grad(input)) {
  //     for (size_t i = 0; i < input.size(); ++i)
  //       input[i] = input[i].transpose(-2, -1).contiguous();
  //     auto input_grad = group_op.call(input, grad_outs);
  //     return {input_grad, other_grad};
  //   } else {
  //     return other_grad;
  //   }
  // }
};

// Performs matrix multiplication according to segments.
class SegmentMatmul : public torch::autograd::Function<SegmentMatmul> {
  // TODO (matthias) Add TensorArg definitions.
 public:
  static variable_list forward(AutogradContext* ctx,
                               Variable input,
                               const at::Tensor& ptr,
                               Variable other) {
    auto out = segment_op.call(input, ptr, other);
    ctx->save_for_backward({input, ptr, other});
    return {out, Variable()};
  }

  static variable_list backward(AutogradContext* ctx, Variable grad_out) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto ptr = saved[1];
    auto other = saved[2].transpose(-2, -1).contiguous();
    auto other_grad = segment_op.call(grad_out, ptr, other);
    if (torch::autograd::any_variable_requires_grad({input})) {
      input = input.transpose(-2, -1).contiguous();
      auto input_grad = segment_op.call(input, ptr, grad_out);
      return {input_grad, other_grad};
    } else {
      return {Variable(), other_grad};
    }
  }
};

std::vector<at::Tensor> grouped_matmul(const std::vector<at::Tensor>& input,
                                       const std::vector<at::Tensor>& other) {
  return GroupedMatmul::apply(input, other);
}

at::Tensor segment_matmul(const at::Tensor& input,
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
