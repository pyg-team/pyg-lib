#include <ATen/ATen.h>
#include <torch/library.h>

#include "pyg_lib/csrc/utils/convert.h"

#include <iostream>

namespace pyg {
namespace ops {

namespace {

void grouped_matmul_out_kernel(const std::vector<at::Tensor>& input,
                               const std::vector<at::Tensor>& other,
                               std::vector<at::Tensor>& out) {
  TORCH_CHECK(input.size() == other.size() && other.size() == out.size(),
              "Size of all input vectors should be equal.");
  for (size_t i = 0; i < out.size(); ++i) {
    TORCH_CHECK(input[i].is_cpu() && other[i].is_cpu() && out[i].is_cpu(),
                "All tensors should be associated with the 'CPU' device.");
  }

  for (size_t i = 0; i < out.size(); ++i)
    at::matmul_out(out[i], input[i], other[i]);
}

std::vector<at::Tensor> grouped_matmul_kernel(
    const std::vector<at::Tensor>& input,
    const std::vector<at::Tensor>& other) {
  std::vector<at::Tensor> out(input.size());
  for (size_t i = 0; i < input.size(); ++i)
    out[i] = input[i].new_empty({input[i].size(0), other[i].size(-1)});

  grouped_matmul_out_kernel(input, other, out);

  return out;
}

at::Tensor segment_matmul_kernel(const at::Tensor& input,
                                 const at::Tensor& ptr,
                                 const at::Tensor& other) {
  const auto size = pyg::utils::size_from_ptr(ptr).cpu();
  const auto sizes = at::IntArrayRef(size.data_ptr<int64_t>(), size.numel());
  const auto out = input.new_empty({input.size(0), other.size(-1)});
  auto out_parts = out.split_with_sizes(/*split_size=*/sizes, /*dim=*/0);
  for (auto& out_part : out_parts)
    out_part.resize_(0);

  grouped_matmul_out_kernel(
      input.contiguous().split_with_sizes(/*split_size=*/sizes, /*dim=*/0),
      other.contiguous().split(/*split_size=*/1, /*dim=*/0), out_parts);

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::grouped_matmul"),
         TORCH_FN(grouped_matmul_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_matmul"),
         TORCH_FN(segment_matmul_kernel));
}

}  // namespace ops
}  // namespace pyg
