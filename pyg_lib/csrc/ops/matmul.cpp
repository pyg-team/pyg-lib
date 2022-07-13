#include "matmul.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/script.h>

namespace pyg {
namespace ops {

namespace {

std::vector<at::Tensor> _grouped_matmul(const std::vector<at::Tensor>& input,
                                        const std::vector<at::Tensor>& other) {
  // TODO (matthias) Add TensorArg definitions.
  // TODO (matthias) Add dispatcher support.
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::grouped_matmul", "")
                       .typed<decltype(grouped_matmul)>();
  return op.call(input, other);
}

at::Tensor _segment_matmul(const at::Tensor& input,
                           const at::Tensor& ptr,
                           const at::Tensor& other) {
  // TODO (matthias) Add TensorArg definitions.
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::segment_matmul", "")
                       .typed<decltype(segment_matmul)>();
  return op.call(input, ptr, other);
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

std::vector<at::Tensor> break_w_ptr(const at::Tensor& tens,
                                    const at::Tensor& ptr) {
  std::vector<at::Tensor> return_list;
  for (size_t i = 0; i < ptr.numel(); ++i)
    return_list.push_back(tens.slice(0, ptr[i - 1], ptr[i]));
  return return_list;
}

at::Tensor reflatten(std::vector<at::Tensor> list) {
  at::Tensor return_tens;
  int ptr = 0;
  for (size_t i = 0; i < list.size(); ++i){
    auto tens_to_store = list[i];
    return_tens.slice(0, ptr, ptr+tens_to_store.size(0)) = tens_to_store;
  }
  return return_tens;
}

// Performs matrix multiplication according to segments.
class SegmentMatmul : public torch::autograd::Function<SegmentMatmul> {
 public:
  static variable_list forward(AutogradContext* ctx,
                               Variable input,
                               Variable ptr,
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
      auto other_t = other.transpose(-2, -1).contiguous();
      input_grad = _segment_matmul(grad_out, ptr, other_t);
    }
    if (torch::autograd::any_variable_requires_grad({other})) {
      variable_list grad_out_list = break_w_ptr(grad_out, ptr);
      variable_list other_list = break_w_ptr(other, ptr);
      other_grad = reflatten(_grouped_matmul(grad_out_list, other_list));
    }
    return {input_grad, Variable(), other_grad};
  }
};

}  // namespace

// Performs matrix multiplication across list of elements.
std::vector<at::Tensor> grouped_matmul(const std::vector<at::Tensor>& input,
                                       const std::vector<at::Tensor>& other) {
  // TODO (matthias) Add autograd support.
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
