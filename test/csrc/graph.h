#pragma once

#include <torch/torch.h>

std::tuple<torch::Tensor, torch::Tensor> cycle_graph(
    int64_t num_nodes,
    torch::TensorOptions options) {
  const auto rowptr = torch::arange(0, 2 * num_nodes + 1, 2, options);
  const auto col1 = torch::arange(-1, num_nodes - 1, options) % num_nodes;
  const auto col2 = torch::arange(1, num_nodes + 1, options) % num_nodes;
  const auto col = torch::stack({col1, col2}, /*dim=*/1).flatten();

  return std::make_tuple(rowptr, col);
}

std::tuple<torch::Tensor, torch::Tensor> cycle_graph(int64_t num_nodes) {
  const auto options = torch::TensorOptions().dtype(torch::kInt64);
  return cycle_graph(num_nodes, options);
}
