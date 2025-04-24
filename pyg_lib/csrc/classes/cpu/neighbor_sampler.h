#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>
#include "pyg_lib/csrc/utils/types.h"

namespace pyg {
namespace classes {

struct HeteroNeighborSampler : torch::CustomClassHolder {
  HeteroNeighborSampler(
      const std::vector<node_type> node_types,
      const std::vector<edge_type> edge_types,
      const c10::Dict<rel_type, at::Tensor> rowptr,
      const c10::Dict<rel_type, at::Tensor> col,
      const c10::optional<c10::Dict<rel_type, at::Tensor>> edge_weight,
      const c10::optional<c10::Dict<node_type, at::Tensor>> node_time,
      const c10::optional<c10::Dict<rel_type, at::Tensor>> edge_time);

  std::tuple<c10::Dict<rel_type, at::Tensor>,                  // row
             c10::Dict<rel_type, at::Tensor>,                  // col
             c10::Dict<node_type, at::Tensor>,                 // node_id
             c10::optional<c10::Dict<rel_type, at::Tensor>>,   // edge_id
             c10::optional<c10::Dict<node_type, at::Tensor>>,  // batch
             c10::Dict<node_type, std::vector<int64_t>>,       // num_sampled_nodes
             c10::Dict<rel_type, std::vector<int64_t>>>        // num_sampled_edges
  sample(const c10::Dict<rel_type, std::vector<int64_t>>& num_neighbors,
         const c10::Dict<node_type, at::Tensor>& seed_node,
         const c10::optional<c10::Dict<node_type, at::Tensor>>& seed_time,
         bool disjoint = false,
         std::string temporal_strategy = "uniform",
         bool return_edge_id = true);
};

}  // namespace classes
}  // namespace pyg

