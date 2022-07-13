#include "matmul.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/script.h>

namespace pyg {
namespace ops {

namespace {

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;
std::vector<at::Tensor> _grouped_matmul(const std::vector<at::Tensor>& input,
                                        const std::vector<at::Tensor>& other) {
  // TODO (matthias) Add TensorArg definitions.
  // TODO (matthias) Add dispatcher support.
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::grouped_matmul", "")
                       .typed<decltype(grouped_matmul)>();
  return op.call(input, other);
}

std::vector<at::Tensor> concat(const std::vector<at::Tensor>& t1,
                               const std::vector<at::Tensor>& t2) {
  std::vector<at::Tensor> t3(t1);
  for (size_t i = 0; i < t2.size(); ++i)
    t3.push_back(t2[i]);
  return t3;
}

auto split(
    const std::vector<at::Tensor>& t,
    int split_index) {
  std::vector<at::Tensor> t1(t.begin(), t.begin() + split_index);
  std::vector<at::Tensor> t2(t.begin() + split_index, t.end());
  return std::make_tuple(t1, t2);
}

class GroupedMatmul : public torch::autograd::Function<GroupedMatmul> {
  // TODO (matthias) Add TensorArg definitions.
 public:
  static variable_list forward(AutogradContext* ctx,
                               std::vector<Variable> input,
                               std::vector<Variable> other) {
    auto out = _grouped_matmul(input, other);
    variable_list input_and_other = concat(input, other);
    ctx->save_for_backward(input_and_other);
    ctx->saved_data["input_len"] = (int)input.size();
    return out;
  }

  static variable_list backward(AutogradContext* ctx, variable_list grad_outs) {
    auto input_and_other = ctx->get_saved_variables();
    int input_len = ctx->saved_data["input_len"].toInt();
    auto input_other_tuple = split(input_and_other, input_len);
    auto input = input_other_tuple[0];
    auto other = input_other_tuple[1];
    variable_list other_t;
    for (size_t i = 0; i < input.size(); ++i)
      other_t.push_back(other[i].transpose(-2, -1));
    auto other_grad = _grouped_matmul(grad_outs, other_t);
    variable_list input_grad;
    // For Simplicity:
    // We assume entire input variable list either requires grad or does not
    if (torch::autograd::any_variable_requires_grad(input)) {
      variable_list input_t;
      for (size_t i = 0; i < input.size(); ++i)
        input_t.push_back(input[i].transpose(-2, -1));
      input_grad = _grouped_matmul(input_t, grad_outs);
    } else {
      for (size_t i = 0; i < input.size(); ++i)
        input_grad.push_back(Variable());
    }
    return concat(input_grad, other_grad);
  }
};

at::Tensor _segment_matmul(const at::Tensor& input,
                           const at::Tensor& ptr,
                           const at::Tensor& other) {
  // TODO (matthias) Add TensorArg definitions.
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::segment_matmul", "")
                       .typed<decltype(segment_matmul)>();
  return op.call(input, ptr, other);
}

std::vector<at::Tensor> break_w_ptr(const at::Tensor& tens,
                                    const at::Tensor& ptr) {
  std::vector<at::Tensor> return_list;
  for (int64_t i = 0; i < ptr.numel(); ++i)
    return_list.push_back(tens.slice(0, ptr.index({i - 1}).item<int>(),
                                     ptr.index({i}).item<int>()));
  return return_list;
}

at::Tensor reflatten(std::vector<at::Tensor> list) {
  at::Tensor return_tens;
  int ptr = 0;
  for (int64_t i = 0; i < list.size(); ++i) {
    auto tens_to_store = list[i];
    return_tens.index_put_(
        {torch::indexing::Slice(ptr, ptr + tens_to_store.size(0))},
        tens_to_store);
    ptr = ptr + tens_to_store.size(0);
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
      auto other_t = other.transpose(-2, -1);
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
  return GroupedMatmul::apply(input, other);
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
