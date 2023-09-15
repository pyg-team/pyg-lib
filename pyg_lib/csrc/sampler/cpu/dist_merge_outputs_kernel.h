#include <ATen/ATen.h>
#include <torch/library.h>
#include "pyg_lib/csrc/utils/types.h"

namespace pyg {
namespace sampler {

std::tuple<at::Tensor,
           c10::optional<at::Tensor>,
           c10::optional<at::Tensor>,
           std::vector<int64_t>>
merge_sampler_outputs_kernel(
    const std::vector<at::Tensor>& nodes,
    const std::vector<std::vector<int64_t>>& cumsum_neighbors_per_node,
    const std::vector<int64_t>& partition_ids,
    const std::vector<int64_t>& partition_orders,
    const int64_t partitions_num,
    const int64_t one_hop_num,
    const c10::optional<std::vector<at::Tensor>>& edge_ids,
    const c10::optional<at::Tensor>& batch,
    bool disjoint,
    bool with_edge);

}  // namespace sampler
}  // namespace pyg
