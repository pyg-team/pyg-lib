#include "matmul.h"
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/script.h>

namespace pyg {
namespace ops {

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

// Performs matrix multiplication across list of elements.
std::vector<at::Tensor> grouped_matmul(const std::vector<at::Tensor>& input,
                                       const std::vector<at::Tensor>& other) {
  // TODO (matthias) Add TensorArg definitions.
  // TODO (matthias) Add autograd support.
  // TODO (matthias) Add dispatcher support.
  // TODO (rishi) Add get GroupedMatmul backward working
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::grouped_matmul", "")
                       .typed<decltype(grouped_matmul)>();
  return op.call(input, other);
}

// static auto group_op = c10::Dispatcher::singleton()
//                            .findSchemaOrThrow("pyg::grouped_matmul", "")
//                            .typed<decltype(grouped_matmul)>();
// class GroupedMatmul : public torch::autograd::Function<GroupedMatmul> {
//   // TODO (matthias) Add TensorArg definitions.
//  public:
//   static std::vector<variable_list> forward(AutogradContext* ctx,
//                                std::vector<Variable> input,
//                                std::vector<Variable> other) {
//     auto out = group_op.call(input, other);
//     ctx->save_for_backward({input, other});
//     return {out};
//   }

//
//   static std::vector<variable_list> backward(AutogradContext* ctx,
//   variable_list grad_outs) {
//     auto saved = ctx->get_saved_variables();
//     variable_list input = saved[0];
//     variable_list other = saved[1];
//     for (size_t i = 0; i < input.size(); ++i)
//       other[i] = other[i].transpose(-2, -1).contiguous();
//     auto other_grad = group_op.call(grad_outs, other);
//     if (torch::autograd::any_variable_requires_grad(input)) {
//       for (size_t i = 0; i < input.size(); ++i)
//         input[i] = input[i].transpose(-2, -1).contiguous();
//       auto input_grad = group_op.call(input, grad_outs);
//       return {input_grad, other_grad};
//     } else {
//       return {variable_list(), other_grad};
//     }
//   }
// };

// std::vector<at::Tensor> grouped_matmul(const std::vector<at::Tensor>& input,
//                                        const std::vector<at::Tensor>& other)
//                                        {
//   return GroupedMatmul::apply(input, other);
// }

static auto segment_op = c10::Dispatcher::singleton()
                             .findSchemaOrThrow("pyg::segment_matmul", "")
                             .typed<decltype(segment_matmul)>();

// Performs matrix multiplication according to segments.
class SegmentMatmul : public torch::autograd::Function<SegmentMatmul> {
  // TODO (matthias) Add TensorArg definitions.
 public:
  static variable_list forward(AutogradContext* ctx,
                               Variable input,
                               const at::Tensor& ptr,
                               Variable other) {
    Variable out = segment_op.call(input, ptr, other)[0];
    ctx->save_for_backward({input, ptr, other});
    return {out};
  }

  static variable_list backward(AutogradContext* ctx, variable_list grad_outs) {
    variable_list saved = ctx->get_saved_variables();
    Variable input = saved[0];
    Variable ptr = saved[1];
    Variable other = saved[2].transpose(-2, -1).contiguous();
    Variable grad_out = grad_outs[0];
    Variable other_grad = segment_op.call(grad_out, ptr, other)[0];
    if (torch::autograd::any_variable_requires_grad({input})) {
      input = input.transpose(-2, -1).contiguous();
      Variable input_grad = segment_op.call(input, ptr, grad_out)[0];
      return {input_grad, other_grad};
    } else {
      return {Variable(), other_grad};
    }
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
}
static auto registry =
    torch::RegisterOperators().op("segment_matmul", &segment_matmul);
}  // namespace ops
}  // namespace pyg
