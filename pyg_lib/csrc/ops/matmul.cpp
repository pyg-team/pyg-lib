#include "matmul.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/script.h>

#include <iostream>
#include "pyg_lib/csrc/utils/convert.h"
namespace pyg {
namespace ops {

namespace {

std::vector<at::Tensor> _grouped_matmul(const std::vector<at::Tensor>& input,
                                        const std::vector<at::Tensor>& other) {
  // TODO (matthias) Add TensorArg definitions.
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

std::vector<at::Tensor> concat(std::vector<at::Tensor> t1,
                               std::vector<at::Tensor> t2) {
  for (size_t i = 0; i < t2.size(); ++i) {
    t1.push_back(t2[i]);
  }
  return t1;
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class GroupedMatmul : public torch::autograd::Function<GroupedMatmul> {
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
    std::vector<at::Tensor> input(input_and_other.begin(),
                                  input_and_other.begin() + input_len);
    std::vector<at::Tensor> other(input_and_other.begin() + input_len,
                                  input_and_other.end());
    for (size_t i = 0; i < input.size(); ++i)
      other[i] = other[i].transpose(-2, -1);
    auto other_grad = _grouped_matmul(grad_outs, other);
    variable_list input_grad;
    // For Simplicity:
    // We assume entire input variable list either requires grad or does not
    if (torch::autograd::any_variable_requires_grad(input)) {
      for (size_t i = 0; i < input.size(); ++i)
        input[i] = input[i].transpose(-2, -1);
      input_grad = _grouped_matmul(input, grad_outs);
    } else {
      for (size_t i = 0; i < input.size(); ++i)
        input_grad.push_back(Variable());
    }
    return concat(input_grad, other_grad);
  }
};

class SegmentMatmul : public torch::autograd::Function<SegmentMatmul> {
 public:
  static variable_list forward(AutogradContext* ctx,
                               Variable input,
                               Variable ptr,
                               Variable other) {
    std::cout << "================= DEBUG =================" << std::endl;
    ctx->save_for_backward({input, ptr, other});
    Variable out = _segment_matmul(input, ptr, other);
    return {out};
  }

  static variable_list backward(AutogradContext* ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto input = saved[0], ptr = saved[1], other = saved[2];

    auto input_grad = Variable();
    std::cout << "================= DEBUG =================" << std::endl;
    std::cout << torch::autograd::any_variable_requires_grad({input});
    if (torch::autograd::any_variable_requires_grad({input})) {
      auto other_t = other.transpose(-2, -1);
      input_grad = _segment_matmul(grad_out, ptr, other_t);
    }

    auto other_grad = Variable();
    std::cout << "================= DEBUG =================" << std::endl;
    std::cout << torch::autograd::any_variable_requires_grad({other});
    if (torch::autograd::any_variable_requires_grad({other})) {
      auto size = pyg::utils::size_from_ptr(ptr).cpu();
      // TODO (matthias) Allow for other types than `int64_t`.
      auto sizes = at::IntArrayRef(size.data_ptr<int64_t>(), size.numel());
      auto input_t = input.transpose(-2, -1);
      auto others_grad = _grouped_matmul(
          input_t.split_with_sizes(/*split_size=*/sizes, /*dim=*/1),
          grad_out.split_with_sizes(/*split_size=*/sizes, /*dim=*/0));
      other_grad = at::stack(others_grad);
    }
    return {input_grad, Variable(), other_grad};
  }
};

}  // namespace

// Performs matrix multiplication across list of elements.
std::vector<at::Tensor> my_grouped_matmul(
    const std::vector<at::Tensor>& input,
    const std::vector<at::Tensor>& other) {
  return GroupedMatmul::apply(input, other);
}

// Performs matrix multiplication according to segments.
at::Tensor my_segment_matmul(const at::Tensor& input,
                             const at::Tensor& ptr,
                             const at::Tensor& other) {
  return SegmentMatmul::apply(input, ptr, other)[0];
}

TORCH_LIBRARY_IMPL(pyg, Autograd, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::my_grouped_matmul"),
         TORCH_FN(my_grouped_matmul));
  m.impl(TORCH_SELECTIVE_NAME("pyg::my_segment_matmul"),
         TORCH_FN(my_segment_matmul));
}

}  // namespace ops
}  // namespace pyg
