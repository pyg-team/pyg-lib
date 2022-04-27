#pragma once

#include <ATen/ATen.h>

std::tuple<at::Tensor, at::Tensor> cycle_graph(int64_t num_nodes,
                                               at::TensorOptions options) {
  const auto rowptr = at::arange(0, 2 * num_nodes + 1, 2, options);
  const auto col1 = at::arange(-1, num_nodes - 1, options) % num_nodes;
  const auto col2 = at::arange(1, num_nodes + 1, options) % num_nodes;
  const auto col = at::stack({col1, col2}, /*dim=*/1).flatten();

  return std::make_tuple(rowptr, col);
}

std::tuple<at::Tensor, at::Tensor> cycle_graph(int64_t num_nodes) {
  const auto options = at::TensorOptions().dtype(at::kLong);
  return cycle_graph(num_nodes, options);
}
