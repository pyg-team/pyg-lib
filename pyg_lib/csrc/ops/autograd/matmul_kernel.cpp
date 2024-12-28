#include "../matmul.h"

#include <torch/autograd.h>

#include "pyg_lib/csrc/utils/convert.h"

namespace pyg {
namespace ops {

namespace {

using torch::autograd::variable_list;

std::vector<at::Tensor> concat(std::vector<at::Tensor> t1,
                               std::vector<at::Tensor> t2) {
  for (size_t i = 0; i < t2.size(); ++i) {
    t1.push_back(t2[i]);
  }
  return t1;
}

class GroupedMatmul : public torch::autograd::Function<GroupedMatmul> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const variable_list input,
                               const variable_list other) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto out = grouped_matmul(input, other);
    variable_list input_and_other = concat(input, other);
    ctx->save_for_backward(input_and_other);
    return out;
  }

  static variable_list backward(torch::autograd::AutogradContext* ctx,
                                variable_list grad_outs) {
    auto input_and_other = ctx->get_saved_variables();
    int input_len = input_and_other.size() / 2;
    variable_list input(input_and_other.begin(),
                        input_and_other.begin() + input_len);
    variable_list other(input_and_other.begin() + input_len,
                        input_and_other.end());

    // We assume entire input variable list either requires grad or does not:
    variable_list other_grad;
    if (torch::autograd::any_variable_requires_grad(other)) {
      for (size_t i = 0; i < input.size(); ++i) {
        other[i] = other[i].transpose(-2, -1);
        other_grad.push_back(torch::matmul(grad_outs[i], other[i]));
      }
    } else {
      for (size_t i = 0; i < other.size(); ++i)
        other_grad.push_back(at::Tensor());
    }

    variable_list input_grad;
    if (torch::autograd::any_variable_requires_grad(input)) {
      for (size_t i = 0; i < input.size(); ++i)
        input[i] = input[i].transpose(-2, -1);
      input_grad = grouped_matmul(input, grad_outs);
    } else {
      for (size_t i = 0; i < input.size(); ++i)
        input_grad.push_back(at::Tensor());
    }
    return concat(input_grad, other_grad);
  }
};

class SegmentMatmul : public torch::autograd::Function<SegmentMatmul> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input,
                               const at::Tensor& ptr,
                               const at::Tensor& other) {
    at::AutoDispatchBelowADInplaceOrView g;
    at::Tensor out = segment_matmul(input, ptr, other);
    ctx->save_for_backward({input, ptr, other});
    return {out};
  }

  static variable_list backward(torch::autograd::AutogradContext* ctx,
                                variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto input = saved[0], ptr = saved[1], other = saved[2];

    auto input_grad = at::Tensor();
    if (torch::autograd::any_variable_requires_grad({input})) {
      auto other_t = other.transpose(-2, -1);
      input_grad = segment_matmul(grad_out, ptr, other_t);
    }

    auto other_grad = at::Tensor();
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

    return {input_grad, at::Tensor(), other_grad};
  }
};

at::Tensor segment_matmul_autograd(const at::Tensor& input,
                                   const at::Tensor& ptr,
                                   const at::Tensor& other) {
  return SegmentMatmul::apply(input, ptr, other)[0];
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, Autograd, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_matmul"),
         TORCH_FN(segment_matmul_autograd));
}

}  // namespace ops
}  // namespace pyg
