#include <ATen/ATen.h>
#include <torch/library.h>
#include "pyg_lib/csrc/utils/types.h"

namespace pyg {
namespace sampler {

std::tuple<at::Tensor, at::Tensor> relabel_neighborhood_kernel(
    const at::Tensor& seed,
    const at::Tensor& sampled_nodes_with_dupl,
    const std::vector<int64_t>& sampled_nbrs_per_node,
    const int64_t num_nodes,
    const c10::optional<at::Tensor>& batch,
    bool csc,
    bool disjoint);

std::tuple<c10::Dict<rel_type, at::Tensor>, c10::Dict<rel_type, at::Tensor>>
hetero_relabel_neighborhood_kernel(
    const std::vector<node_type>& node_types,
    const std::vector<edge_type>& edge_types,
    const c10::Dict<node_type, at::Tensor>& seed_dict,
    const c10::Dict<node_type, at::Tensor>& sampled_nodes_with_dupl_dict,
    const c10::Dict<node_type, std::vector<int64_t>>&
        sampled_nbrs_per_node_dict,
    const c10::Dict<node_type, int64_t>& num_nodes_dict,
    const c10::optional<c10::Dict<node_type, at::Tensor>>& batch_dict,
    bool csc,
    bool disjoint);

}  // namespace sampler
}  // namespace pyg
