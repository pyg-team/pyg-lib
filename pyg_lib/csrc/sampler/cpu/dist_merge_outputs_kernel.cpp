#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <parallel_hashmap/phmap.h>
#include <torch/library.h>

#include "pyg_lib/csrc/sampler/cpu/mapper.h"
#include "pyg_lib/csrc/utils/cpu/convert.h"
#include "pyg_lib/csrc/utils/types.h"

namespace pyg {
namespace sampler {

namespace {

template <bool disjoint>
std::tuple<at::Tensor,
           at::Tensor,
           c10::optional<at::Tensor>,
           std::vector<int64_t>>
merge_outputs(
    const std::vector<at::Tensor>& node_ids,
    const std::vector<at::Tensor>& edge_ids,
    const std::vector<std::vector<int64_t>>& cumsum_neighbors_per_node,
    const std::vector<int64_t>& partition_ids,
    const std::vector<int64_t>& partition_orders,
    const int64_t num_partitions,
    const int64_t num_neighbors,
    const c10::optional<at::Tensor>& batch) {
  at::Tensor out_node_id;
  at::Tensor out_edge_id;
  c10::optional<at::Tensor> out_batch = c10::nullopt;

  auto offset = num_neighbors;

  if (num_neighbors < 0) {
    // find maximum population
    std::vector<std::vector<int64_t>> population(num_partitions);
    std::vector<int64_t> max_populations(num_partitions);

    at::parallel_for(0, num_partitions, 1, [&](size_t _s, size_t _e) {
      for (auto p_id = _s; p_id < _e; p_id++) {
        auto cummsum1 =
            std::vector<int64_t>(cumsum_neighbors_per_node[p_id].begin() + 1,
                                 cumsum_neighbors_per_node[p_id].end());
        auto cummsum2 =
            std::vector<int64_t>(cumsum_neighbors_per_node[p_id].begin(),
                                 cumsum_neighbors_per_node[p_id].end() - 1);
        std::transform(cummsum1.begin(), cummsum1.end(), cummsum2.begin(),
                       std::back_inserter(population[p_id]),
                       [](int64_t a, int64_t b) { return std::abs(a - b); });
        auto max =
            *max_element(population[p_id].begin(), population[p_id].end());
        max_populations[p_id] = max;
      }
    });
    offset = *max_element(max_populations.begin(), max_populations.end());
  }

  const auto p_size = partition_ids.size();
  std::vector<int64_t> num_sampled_neighbors_per_node(p_size);

  const auto scalar_type = node_ids[0].scalar_type();
  AT_DISPATCH_INTEGRAL_TYPES(scalar_type, "merge_outputs_kernel", [&] {
    std::vector<scalar_t> sampled_node_ids(p_size * offset, -1);
    std::vector<scalar_t> sampled_edge_ids(p_size * offset, -1);
    std::vector<std::vector<scalar_t>> sampled_node_ids_vec(p_size);
    std::vector<std::vector<scalar_t>> sampled_edge_ids_vec(p_size);

    std::vector<scalar_t> sampled_batch;
    if constexpr (disjoint) {
      sampled_batch = std::vector<scalar_t>(p_size * offset, -1);
    }
    const auto batch_data =
        disjoint ? batch.value().data_ptr<scalar_t>() : nullptr;

    for (auto p_id = 0; p_id < num_partitions; p_id++) {
      sampled_node_ids_vec[p_id] =
          pyg::utils::to_vector<scalar_t>(node_ids[p_id]);
      sampled_edge_ids_vec[p_id] =
          pyg::utils::to_vector<scalar_t>(edge_ids[p_id]);
    }
    at::parallel_for(0, p_size, 1, [&](size_t _s, size_t _e) {
      for (auto j = _s; j < _e; j++) {
        auto p_id = partition_ids[j];
        auto p_order = partition_orders[j];

        // When it comes to node and batch, we omit seed nodes.
        // In the case of edges, we take into account all sampled edge ids.
        auto begin_node = cumsum_neighbors_per_node[p_id][p_order];
        auto begin_edge = begin_node - cumsum_neighbors_per_node[p_id][0];

        auto end_node = cumsum_neighbors_per_node[p_id][p_order + 1];
        auto end_edge = end_node - cumsum_neighbors_per_node[p_id][0];

        std::copy(sampled_node_ids_vec[p_id].begin() + begin_node,
                  sampled_node_ids_vec[p_id].begin() + end_node,
                  sampled_node_ids.begin() + j * offset);
        std::copy(sampled_edge_ids_vec[p_id].begin() + begin_edge,
                  sampled_edge_ids_vec[p_id].begin() + end_edge,
                  sampled_edge_ids.begin() + j * offset);

        if constexpr (disjoint) {
          std::fill(sampled_batch.begin() + j * offset,
                    sampled_batch.begin() + j * offset + end_node - begin_node,
                    batch_data[j]);
        }

        num_sampled_neighbors_per_node[j] = end_node - begin_node;
      }
    });

    // Remove auxilary -1 numbers:
    auto neg =
        std::remove(sampled_node_ids.begin(), sampled_node_ids.end(), -1);
    sampled_node_ids.erase(neg, sampled_node_ids.end());
    out_node_id = pyg::utils::from_vector(sampled_node_ids);

    neg = std::remove(sampled_edge_ids.begin(), sampled_edge_ids.end(), -1);
    sampled_edge_ids.erase(neg, sampled_edge_ids.end());
    out_edge_id = pyg::utils::from_vector(sampled_edge_ids);

    if constexpr (disjoint) {
      neg = std::remove(sampled_batch.begin(), sampled_batch.end(), -1);
      sampled_batch.erase(neg, sampled_batch.end());
      out_batch = pyg::utils::from_vector(sampled_batch);
    }
  });

  return std::make_tuple(out_node_id, out_edge_id, out_batch,
                         num_sampled_neighbors_per_node);
}

#define DISPATCH_MERGE_OUTPUTS(disjoint, ...) \
  if (disjoint)                               \
    return merge_outputs<true>(__VA_ARGS__);  \
  if (!disjoint)                              \
    return merge_outputs<false>(__VA_ARGS__);

}  // namespace

std::tuple<at::Tensor,
           at::Tensor,
           c10::optional<at::Tensor>,
           std::vector<int64_t>>
merge_sampler_outputs_kernel(
    const std::vector<at::Tensor>& node_ids,
    const std::vector<at::Tensor>& edge_ids,
    const std::vector<std::vector<int64_t>>& cumsum_neighbors_per_node,
    const std::vector<int64_t>& partition_ids,
    const std::vector<int64_t>& partition_orders,
    const int64_t num_partitions,
    const int64_t num_neighbors,
    const c10::optional<at::Tensor>& batch,
    bool disjoint) {
  DISPATCH_MERGE_OUTPUTS(
      disjoint, node_ids, edge_ids, cumsum_neighbors_per_node, partition_ids,
      partition_orders, num_partitions, num_neighbors, batch);
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
