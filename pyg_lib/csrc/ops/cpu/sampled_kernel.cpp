#include <ATen/ATen.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

namespace {

enum FnType { ADD, SUB, MUL, DIV };
const std::map<std::string, FnType> to_fn_type = {
    {"add", ADD},
    {"sub", SUB},
    {"mul", MUL},
    {"div", DIV},
};

at::Tensor sampled_op_kernel(const at::Tensor& left,
                             const at::Tensor& right,
                             const at::optional<at::Tensor> left_index,
                             const at::optional<at::Tensor> right_index,
                             const std::string fn) {
  auto a = left;
  if (left_index.has_value()) {
    a = left.index_select(0, left_index.value());
  }

  auto b = right;
  if (right_index.has_value()) {
    b = right.index_select(0, right_index.value());
  }

  auto fn_type = to_fn_type.at(fn);

  at::Tensor out;
  if (fn_type == ADD) {
    out = a + b;
  } else if (fn_type == SUB) {
    out = a - b;
  } else if (fn_type == MUL) {
    out = a * b;
  } else if (fn_type == DIV) {
    out = a / b;
  }

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::sampled_op"), TORCH_FN(sampled_op_kernel));
}

}  // namespace ops
}  // namespace pyg
