#include "dist_neighbor.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

#include "pyg_lib/csrc/utils/check.h"

namespace pyg {
namespace sampler {

std::tuple<at::Tensor, at::Tensor> relabel_neighborhood(
    const at::Tensor& seed,
    const at::Tensor& sampled_nodes_with_dupl,
    const std::vector<int64_t>& sampled_nbrs_per_node,
    const int64_t num_nodes,
    const c10::optional<at::Tensor>& batch,
    bool csc,
    bool disjoint) {
  at::TensorArg seed_args{seed, "seed", 1};
  at::TensorArg sampled_nodes_with_dupl_args{sampled_nodes_with_dupl,
                                             "sampled_nodes_with_dupl", 1};

  at::CheckedFrom c = "relabel_neighborhood";
  at::checkAllDefined(c, {sampled_nodes_with_dupl_args, seed_args});
  at::checkAllSameType(c, {sampled_nodes_with_dupl_args, seed_args});

  TORCH_CHECK(seed.is_contiguous(), "Non-contiguous 'seed'");
  TORCH_CHECK(sampled_nodes_with_dupl.is_contiguous(),
              "Non-contiguous 'sampled_nodes_with_dupl'");

  if (disjoint) {
    TORCH_CHECK(batch.has_value(),
                "Batch needs to be specified to create disjoint subgraphs");
    TORCH_CHECK(batch.value().is_contiguous(), "Non-contiguous 'batch'");
    TORCH_CHECK(batch.value().numel() == sampled_nodes_with_dupl.numel(),
                "Each node must belong to a subgraph.'");
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::relabel_neighborhood", "")
                       .typed<decltype(relabel_neighborhood)>();
  return op.call(seed, sampled_nodes_with_dupl, sampled_nbrs_per_node,
                 num_nodes, batch, csc, disjoint);
}

std::tuple<c10::Dict<rel_type, at::Tensor>, c10::Dict<rel_type, at::Tensor>>
hetero_relabel_neighborhood(
    const std::vector<node_type>& node_types,
    const std::vector<edge_type>& edge_types,
    const c10::Dict<node_type, at::Tensor>& seed_dict,
    const c10::Dict<node_type, at::Tensor>& sampled_nodes_with_dupl_dict,
    const c10::Dict<node_type, std::vector<int64_t>>&
        sampled_nbrs_per_node_dict,
    const c10::Dict<node_type, int64_t>& num_nodes_dict,
    const c10::optional<c10::Dict<node_type, at::Tensor>>& batch_dict,
    bool csc,
    bool disjoint) {
  std::vector<at::TensorArg> seed_dict_args;
  std::vector<at::TensorArg> sampled_nodes_with_dupl_dict_args;
  pyg::utils::fill_tensor_args(seed_dict_args, seed_dict, "seed_dict", 0);
  pyg::utils::fill_tensor_args(sampled_nodes_with_dupl_dict_args,
                               sampled_nodes_with_dupl_dict,
                               "sampled_nodes_with_dupl_dict", 0);
  at::CheckedFrom c{"hetero_relabel_neighborhood"};

  at::checkAllDefined(c, seed_dict_args);
  at::checkAllDefined(c, sampled_nodes_with_dupl_dict_args);
  at::checkSameType(c, seed_dict_args[0], sampled_nodes_with_dupl_dict_args[0]);

  for (const auto& kv : seed_dict) {
    const at::Tensor& seed = kv.value();
    TORCH_CHECK(seed.is_contiguous(), "Non-contiguous 'seed'");
  }
  for (const auto& kv : sampled_nodes_with_dupl_dict) {
    const at::Tensor& sampled_nodes_with_dupl = kv.value();
    TORCH_CHECK(sampled_nodes_with_dupl.is_contiguous(),
                "Non-contiguous 'sampled_nodes_with_dupl'");
  }

  if (disjoint) {
    TORCH_CHECK(batch_dict.has_value(),
                "Batch needs to be specified to create disjoint subgraphs");
    for (const auto& kv : batch_dict.value()) {
      const at::Tensor& batch = kv.value();
      const at::Tensor& sampled_nodes_with_dupl =
          sampled_nodes_with_dupl_dict.at(kv.key());
      TORCH_CHECK(batch.is_contiguous(), "Non-contiguous 'batch'");
      TORCH_CHECK(batch.numel() == sampled_nodes_with_dupl.numel(),
                  "Each node must belong to a subgraph.'");
    }
  }

  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("pyg::hetero_relabel_neighborhood", "")
          .typed<decltype(hetero_relabel_neighborhood)>();
  return op.call(node_types, edge_types, seed_dict,
                 sampled_nodes_with_dupl_dict, sampled_nbrs_per_node_dict,
                 num_nodes_dict, batch_dict, csc, disjoint);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(
      TORCH_SELECTIVE_SCHEMA("pyg::relabel_neighborhood(Tensor seed, Tensor "
                             "sampled_nodes_with_dupl, int[] "
                             "sampled_nbrs_per_node, int num_nodes, Tensor? "
                             "batch = None, bool csc = False, bool "
                             "disjoint = False) "
                             "-> (Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::hetero_relabel_neighborhood(str[] node_types, (str, str, str)[] "
      "edge_types, Dict(str, Tensor) seed_dict, Dict(str, Tensor) "
      "sampled_nodes_with_dupl_dict, Dict(str, int[]) "
      "sampled_nbrs_per_node_dict, Dict(str, int) num_nodes_dict, Dict(str, "
      "Tensor)? batch_dict = None, bool csc = False, bool "
      "disjoint = False) "
      "-> (Dict(str, Tensor), Dict(str, Tensor))"));
}

}  // namespace sampler
}  // namespace pyg
