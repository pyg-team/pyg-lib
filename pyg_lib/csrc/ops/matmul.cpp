#include "matmul.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/script.h>

#include "pyg_lib/csrc/utils/check.h"
#include "pyg_lib/csrc/utils/convert.h"

namespace pyg {
namespace ops {

namespace {

std::vector<at::Tensor> _grouped_matmul(const at::TensorList input,
                                        const at::TensorList other) {
  TORCH_CHECK(input.size() == other.size(),
              "Number of 'input' tensors must match number of 'other' tensors");
  const auto n_tensors = input.size();
  std::vector<at::TensorArg> input_args;
  std::vector<at::TensorArg> other_args;
  pyg::utils::fill_tensor_args(input_args, input, "input", 0);
  pyg::utils::fill_tensor_args(other_args, other, "other", 1);
  at::CheckedFrom c{"grouped_matmul"};

  at::checkAllDefined(c, input_args);
  at::checkAllDefined(c, other_args);
  at::checkAllSameType(c, input_args);
  at::checkAllSameType(c, other_args);
  at::checkSameType(c, input_args[0], other_args[0]);
  for (size_t i = 0; i < n_tensors; ++i) {
    at::checkDim(c, input_args[i], 2);
    at::checkDim(c, other_args[i], 2);
    at::checkSize(c, other_args[i], 0, input_args[i]->size(-1));
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::grouped_matmul", "")
                       .typed<decltype(_grouped_matmul)>();
  return op.call(input, other);
}

at::Tensor _segment_matmul(const at::Tensor& input,
                           const at::Tensor& ptr,
                           const at::Tensor& other) {
  at::TensorArg input_arg{input, "input", 0};
  at::TensorArg ptr_arg{ptr, "ptr", 1};
  at::TensorArg other_arg{other, "other", 2};
  at::CheckedFrom c{"segment_matmul"};

  at::checkAllDefined(c, {input_arg, ptr_arg, other_arg});
  at::checkSameType(c, input_arg, other_arg);
  at::checkDim(c, input_arg, 2);
  at::checkDim(c, ptr_arg, 1);
  at::checkDim(c, other_arg, 3);
  at::checkSize(c, other_arg, 1, input_arg->size(-1));
  at::checkNumel(c, ptr_arg, other_arg->size(0) + 1);

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::segment_matmul", "")
                       .typed<decltype(_segment_matmul)>();
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
                               variable_list input,
                               variable_list other) {
    auto out = _grouped_matmul(input, other);
    variable_list input_and_other = concat(input, other);
    ctx->save_for_backward(input_and_other);
    ctx->saved_data["input_len"] = (int)input.size();
    return out;
  }

  static variable_list backward(AutogradContext* ctx, variable_list grad_outs) {
    auto input_and_other = ctx->get_saved_variables();
    int input_len = ctx->saved_data["input_len"].toInt();
    variable_list input(input_and_other.begin(),
                        input_and_other.begin() + input_len);
    variable_list other(input_and_other.begin() + input_len,
                        input_and_other.end());
    variable_list other_grad;
    // For Simplicity:
    // We assume entire input variable list either requires grad or does not
    if (torch::autograd::any_variable_requires_grad(other)) {
      for (size_t i = 0; i < input.size(); ++i) {
        other[i] = other[i].transpose(-2, -1);
        other_grad.push_back(torch::matmul(grad_outs[i], other[i]));
      }
    } else {
      for (size_t i = 0; i < other.size(); ++i)
        other_grad.push_back(Variable());
    }

    variable_list input_grad;
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
                               at::Tensor ptr,
                               Variable other) {
    Variable out = _segment_matmul(input, ptr, other);
    ctx->save_for_backward({input, ptr, other});
    return {out};
  }

  static variable_list backward(AutogradContext* ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto input = saved[0], ptr = saved[1], other = saved[2];

    auto input_grad = Variable();
    if (torch::autograd::any_variable_requires_grad({input})) {
      auto other_t = other.transpose(-2, -1);
      input_grad = _segment_matmul(grad_out, ptr, other_t);
    }

    auto other_grad = Variable();
    if (torch::autograd::any_variable_requires_grad({other})) {
      auto size = pyg::utils::size_from_ptr(ptr).cpu();
      // TODO (matthias) Allow for other types than `int64_t`.
      auto sizes = at::IntArrayRef(size.data_ptr<int64_t>(), size.numel());
      auto input_t = input.transpose(-2, -1);
      variable_list split_input_t =
          input_t.split_with_sizes(/*split_size=*/sizes, /*dim=*/1);
      variable_list grad_out_split =
          grad_out.split_with_sizes(/*split_size=*/sizes, /*dim=*/0);
      variable_list others_grad;
      for (size_t i = 0; i < split_input_t.size(); ++i)
        others_grad.push_back(
            torch::matmul(split_input_t[i], grad_out_split[i]));
      other_grad = at::stack(others_grad);
    }

    return {input_grad, Variable(), other_grad};
  }
};

}  // namespace

// Performs matrix multiplication across list of elements.
std::vector<at::Tensor> grouped_matmul_autograd(const variable_list input,
                                                const variable_list other) {
  return GroupedMatmul::apply(input, other);
  // return _grouped_matmul(input, other);
}

// Performs matrix multiplication according to segments.
at::Tensor segment_matmul_autograd(const Variable input,
                                   const at::Tensor& ptr,
                                   const Variable other) {
  return SegmentMatmul::apply(input, ptr, other)[0];
  // return _segment_matmul(input, ptr, other);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(
      TORCH_SELECTIVE_SCHEMA("pyg::grouped_matmul_autograd(Tensor[] input, "
                             "Tensor[] other) -> Tensor[]"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::segment_matmul_autograd(Tensor input, Tensor ptr, "
      "Tensor other) -> Tensor"));
}

}  // namespace ops
}  // namespace pyg
