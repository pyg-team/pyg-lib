#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "parallel_hashmap/phmap.h"

#include "pyg_lib/csrc/sampler/cpu/mapper.h"
#include "pyg_lib/csrc/utils/cpu/convert.h"
#include "pyg_lib/csrc/utils/types.h"

namespace pyg {
namespace sampler {

namespace {

template <bool disjoint, bool with_edge>
std::tuple<at::Tensor,
           c10::optional<at::Tensor>,
           c10::optional<at::Tensor>,
           std::vector<int64_t>>
merge_outputs(
    const std::vector<at::Tensor>& nodes,
    const std::vector<std::vector<int64_t>>& cumm_sampled_nbrs_per_node,
    const std::vector<int64_t>& partition_ids,
    const std::vector<int64_t>& partition_orders,
    const int64_t partitions_num,
    const int64_t one_hop_num,
    const c10::optional<std::vector<at::Tensor>>& edge_ids,
    const c10::optional<std::vector<at::Tensor>>& batch) {
  at::Tensor out_node;
  c10::optional<at::Tensor> out_edge_id = c10::nullopt;
  c10::optional<at::Tensor> out_batch = c10::nullopt;
  int64_t offset = one_hop_num;

  if (one_hop_num < 0) {
    // find maximum population
    std::vector<int64_t> population;
    std::vector<int64_t> max_populations(partitions_num);

    at::parallel_for(0, partitions_num, 1, [&](size_t _s, size_t _e) {
      for (auto p_id = _s; p_id < _e; p_id++) {
        auto cummsum1 =
            std::vector<int64_t>(cumm_sampled_nbrs_per_node[p_id].begin() + 1,
                                 cumm_sampled_nbrs_per_node[p_id].end());
        auto cummsum2 =
            std::vector<int64_t>(cumm_sampled_nbrs_per_node[p_id].begin(),
                                 cumm_sampled_nbrs_per_node[p_id].end() - 1);
        std::transform(cummsum1.begin(), cummsum1.end(), cummsum2.begin(),
                       std::back_inserter(population),
                       [](int64_t a, int64_t b) { return std::abs(a - b); });
        auto max = *max_element(population.begin(), population.end());
        max_populations[p_id] = max;
      }
    });
    offset = *max_element(max_populations.begin(), max_populations.end());
  }

  const auto p_size = partition_ids.size();
  std::vector<int64_t> sampled_nbrs_per_node(p_size);

  const auto scalar_type = nodes[0].scalar_type();
  AT_DISPATCH_INTEGRAL_TYPES(scalar_type, "merge_outputs_kernel", [&] {
    std::vector<scalar_t> sampled_nodes(p_size * offset, -1);
    std::vector<scalar_t> sampled_edge_ids;
    std::vector<scalar_t> sampled_batch;
    std::vector<std::vector<scalar_t>> sampled_nodes_vec(p_size);
    std::vector<std::vector<scalar_t>> edge_ids_vec;
    std::vector<std::vector<scalar_t>> batch_vec(p_size);

    if constexpr (with_edge) {
      sampled_edge_ids = std::vector<scalar_t>(p_size * offset, -1);
      edge_ids_vec = std::vector<std::vector<scalar_t>>(p_size);
    }
    if constexpr (disjoint) {
      sampled_batch = std::vector<scalar_t>(p_size * offset, -1);
      batch_vec = std::vector<std::vector<scalar_t>>(p_size);
    }

    for (auto p_id = 0; p_id < partitions_num; p_id++) {
      sampled_nodes_vec[p_id] = pyg::utils::to_vector<scalar_t>(nodes[p_id]);

      if constexpr (with_edge)
        edge_ids_vec[p_id] =
            pyg::utils::to_vector<scalar_t>(edge_ids.value()[p_id]);

      if constexpr (disjoint)
        batch_vec[p_id] = pyg::utils::to_vector<scalar_t>(batch.value()[p_id]);
    }
    at::parallel_for(0, p_size, 1, [&](size_t _s, size_t _e) {
      for (auto j = _s; j < _e; j++) {
        auto p_id = partition_ids[j];
        auto p_order = partition_orders[j];

        // When it comes to node and batch, we omit seed nodes.
        // In the case of edges, we take into account all sampled edge ids.
        auto begin = cumm_sampled_nbrs_per_node[p_id][p_order];
        auto begin_edge = begin - cumm_sampled_nbrs_per_node[p_id][0];

        auto end = cumm_sampled_nbrs_per_node[p_id][p_order + 1];
        auto end_edge = end - cumm_sampled_nbrs_per_node[p_id][0];

        std::copy(sampled_nodes_vec[p_id].begin() + begin,
                  sampled_nodes_vec[p_id].begin() + end,
                  sampled_nodes.begin() + j * offset);
        if constexpr (with_edge)
          std::copy(edge_ids_vec[p_id].begin() + begin_edge,
                    edge_ids_vec[p_id].begin() + end_edge,
                    sampled_edge_ids.begin() + j * offset);
        if constexpr (disjoint)
          std::copy(batch_vec[p_id].begin() + begin,
                    batch_vec[p_id].begin() + end,
                    sampled_batch.begin() + j * offset);

        sampled_nbrs_per_node[j] = end - begin;
      }
    });

    // remove auxilary -1 numbers
    auto neg = std::remove(sampled_nodes.begin(), sampled_nodes.end(), -1);
    sampled_nodes.erase(neg, sampled_nodes.end());

    out_node = pyg::utils::from_vector(sampled_nodes);

    if constexpr (with_edge) {
      neg = std::remove(sampled_edge_ids.begin(), sampled_edge_ids.end(), -1);
      sampled_edge_ids.erase(neg, sampled_edge_ids.end());

      out_edge_id = pyg::utils::from_vector(sampled_edge_ids);
    }

    if constexpr (disjoint) {
      neg = std::remove(sampled_batch.begin(), sampled_batch.end(), -1);
      sampled_batch.erase(neg, sampled_batch.end());

      out_batch = pyg::utils::from_vector(sampled_batch);
    }
  });

  return std::make_tuple(out_node, out_edge_id, out_batch,
                         sampled_nbrs_per_node);
}

#define DISPATCH_MERGE_OUTPUTS(disjoint, with_edge, ...) \
  if (disjoint && with_edge)                             \
    return merge_outputs<true, true>(__VA_ARGS__);       \
  if (!disjoint && with_edge)                            \
    return merge_outputs<false, true>(__VA_ARGS__);      \
  if (disjoint && !with_edge)                            \
    return merge_outputs<true, false>(__VA_ARGS__);      \
  if (!disjoint && !with_edge)                           \
    return merge_outputs<false, false>(__VA_ARGS__);

}  // namespace

std::tuple<at::Tensor,
           c10::optional<at::Tensor>,
           c10::optional<at::Tensor>,
           std::vector<int64_t>>
merge_sampler_outputs_kernel(
    const std::vector<at::Tensor>& nodes,
    const std::vector<std::vector<int64_t>>& cumm_sampled_nbrs_per_node,
    const std::vector<int64_t>& partition_ids,
    const std::vector<int64_t>& partition_orders,
    const int64_t partitions_num,
    const int64_t one_hop_num,
    const c10::optional<std::vector<at::Tensor>>& edge_ids,
    const c10::optional<std::vector<at::Tensor>>& batch,
    bool disjoint,
    bool with_edge) {
  DISPATCH_MERGE_OUTPUTS(disjoint, with_edge, nodes, cumm_sampled_nbrs_per_node,
                         partition_ids, partition_orders, partitions_num,
                         one_hop_num, edge_ids, batch);
}

// We use `BackendSelect` as a fallback to the dispatcher logic as automatic
// dispatching of std::vector<at::Tensor> is not yet supported by PyTorch.
// See: pytorch/aten/src/ATen/templates/RegisterBackendSelect.cpp.
TORCH_LIBRARY_IMPL(pyg, BackendSelect, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::merge_sampler_outputs"),
         TORCH_FN(merge_sampler_outputs_kernel));
}

}  // namespace sampler
}  // namespace pyg
