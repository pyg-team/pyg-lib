#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <random>

#include "pcg_random.hpp"

#include "pyg_lib/csrc/random/cpu/rand_engine.h"
#include "pyg_lib/csrc/sampler/cpu/mapper.h"
#include "pyg_lib/csrc/sampler/cpu/neighbor_kernel.h"
#include "pyg_lib/csrc/sampler/subgraph.h"
#include "pyg_lib/csrc/utils/cpu/convert.h"
#include "pyg_lib/csrc/utils/types.h"

namespace pyg {
namespace sampler {

namespace {

// Helper classes for bipartite labor sampling //////////////////////////////

// `node_t` is either a scalar or a pair of scalars (example_id, node_id):
template <typename node_t, typename scalar_t, bool save_edge_ids>
class LaborSampler {
 public:
  LaborSampler(const scalar_t* rowptr, const scalar_t* col)
      : rowptr_(rowptr), col_(col) {}

  void uniform_sample(const node_t global_src_node,
                      const scalar_t local_src_node,
                      const int64_t count,
                      pyg::sampler::Mapper<node_t, scalar_t>& dst_mapper,
                      const int64_t random_seed,
                      std::vector<node_t>& out_global_dst_nodes,
                      std::vector<std::pair<float, scalar_t>>& heap) {
    const auto row_start = rowptr_[global_src_node];
    const auto row_end = rowptr_[global_src_node + 1];
    _sample(global_src_node, local_src_node, row_start, row_end, count,
            dst_mapper, random_seed, out_global_dst_nodes, heap);
  }

  std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
  get_sampled_edges(bool csc = false) {
    const auto row = pyg::utils::from_vector(sampled_rows_);
    const auto col = pyg::utils::from_vector(sampled_cols_);
    c10::optional<at::Tensor> edge_id = c10::nullopt;
    if (save_edge_ids) {
      edge_id = pyg::utils::from_vector(sampled_edge_ids_);
    }
    if (!csc) {
      return std::make_tuple(row, col, edge_id);
    } else {
      return std::make_tuple(col, row, edge_id);
    }
  }

 private:
  void _sample(const node_t global_src_node,
               const scalar_t local_src_node,
               const scalar_t row_start,
               const scalar_t row_end,
               const int64_t count,
               pyg::sampler::Mapper<node_t, scalar_t>& dst_mapper,
               const int64_t random_seed,
               std::vector<node_t>& out_global_dst_nodes,
               std::vector<std::pair<float, scalar_t>>& heap) {
    if (count == 0)
      return;

    const auto population = row_end - row_start;

    if (population == 0)
      return;

    // Case 1: Sample the full neighborhood:
    if (count < 0 || count >= population) {
      for (scalar_t edge_id = row_start; edge_id < row_end; ++edge_id) {
        add(edge_id, global_src_node, local_src_node, dst_mapper,
            out_global_dst_nodes);
      }
    }
    // Case 2: Sample with sequential poisson sampling:
    else {
      heap.clear();
      for (size_t i = 0; i < count; i++) {
        pcg32 ng(random_seed, col_[row_start + i]);
        std::uniform_real_distribution<float> uni;
        const auto rnd = uni(ng);
        heap.emplace_back(rnd, row_start + i);
      }
      std::make_heap(heap.begin(), heap.end());
      for (size_t i = count; i < population; ++i) {
        pcg32 ng(random_seed, col_[row_start + i]);
        std::uniform_real_distribution<float> uni;
        const auto rnd = uni(ng);
        if (rnd < heap[0].first) {
          std::pop_heap(heap.begin(), heap.end());
          heap.back() = std::make_pair(rnd, row_start + i);
          std::push_heap(heap.begin(), heap.end());
        }
      }
      for (std::size_t i = 0; i < count; ++i) {
        const auto edge_id = heap[i].second;
        add(edge_id, global_src_node, local_src_node, dst_mapper,
            out_global_dst_nodes);
      }
    }
  }

  inline void add(const scalar_t edge_id,
                  const node_t global_src_node,
                  const scalar_t local_src_node,
                  pyg::sampler::Mapper<node_t, scalar_t>& dst_mapper,
                  std::vector<node_t>& out_global_dst_nodes) {
    const auto global_dst_node_value = col_[edge_id];
    const auto res = dst_mapper.insert(global_dst_node_value);
    if (res.second) {  // not yet sampled.
      out_global_dst_nodes.push_back(global_dst_node_value);
    }
    {
      num_sampled_edges_per_hop.back()++;
      sampled_rows_.push_back(local_src_node);
      sampled_cols_.push_back(res.first);
      if (save_edge_ids) {
        sampled_edge_ids_.push_back(edge_id);
      }
    }
  }

  const scalar_t* rowptr_;
  const scalar_t* col_;
  std::vector<scalar_t> sampled_rows_;
  std::vector<scalar_t> sampled_cols_;
  std::vector<scalar_t> sampled_edge_ids_;

 public:
  std::vector<int64_t> num_sampled_edges_per_hop;
};

// Homogeneous labor sampling ///////////////////////////////////////////////

template <bool return_edge_id>
std::tuple<at::Tensor,
           at::Tensor,
           at::Tensor,
           c10::optional<at::Tensor>,
           std::vector<int64_t>,
           std::vector<int64_t>>
sample(const at::Tensor& rowptr,
       const at::Tensor& col,
       const at::Tensor& seed,
       const std::vector<int64_t>& num_neighbors,
       const bool csc,
       const bool layer_dependency,
       const int64_t random_seed) {
  TORCH_CHECK(rowptr.is_contiguous(), "Non-contiguous 'rowptr'");
  TORCH_CHECK(col.is_contiguous(), "Non-contiguous 'col'");
  TORCH_CHECK(seed.is_contiguous(), "Non-contiguous 'seed'");

  at::Tensor out_row, out_col, out_node_id;
  c10::optional<at::Tensor> out_edge_id = c10::nullopt;
  std::vector<int64_t> num_sampled_nodes_per_hop;
  std::vector<int64_t> num_sampled_edges_per_hop;

  AT_DISPATCH_INTEGRAL_TYPES(seed.scalar_type(), "sample_kernel", [&] {
    typedef scalar_t node_t;
    typedef LaborSampler<node_t, scalar_t, return_edge_id> LaborSamplerImpl;

    std::vector<node_t> sampled_nodes;
    auto mapper = Mapper<node_t, scalar_t>(/*num_nodes=*/rowptr.size(0) - 1);
    auto sampler =
        LaborSamplerImpl(rowptr.data_ptr<scalar_t>(), col.data_ptr<scalar_t>());

    const auto seed_data = seed.data_ptr<scalar_t>();
    sampled_nodes = pyg::utils::to_vector<scalar_t>(seed);
    mapper.fill(seed);

    num_sampled_nodes_per_hop.push_back(seed.numel());

    size_t begin = 0, end = seed.size(0);
    const auto max_count =
        *std::max_element(num_neighbors.begin(), num_neighbors.end());
    std::vector<std::pair<float, scalar_t>> heap;
    if (max_count > 0)
      heap.reserve(max_count);
    for (size_t ell = 0; ell < num_neighbors.size(); ++ell) {
      const auto count = num_neighbors[ell];
      sampler.num_sampled_edges_per_hop.push_back(0);
      const int64_t random_seed_ell =
          random_seed + (layer_dependency ? 0 : ell);
      for (size_t i = begin; i < end; ++i) {
        sampler.uniform_sample(/*global_src_node=*/sampled_nodes[i],
                               /*local_src_node=*/i, count, mapper,
                               random_seed_ell,
                               /*out_global_dst_nodes=*/sampled_nodes, heap);
      }
      begin = end, end = sampled_nodes.size();
      num_sampled_nodes_per_hop.push_back(end - begin);
    }

    out_node_id = pyg::utils::from_vector(sampled_nodes);
    {
      std::tie(out_row, out_col, out_edge_id) = sampler.get_sampled_edges(csc);
    }

    num_sampled_edges_per_hop = sampler.num_sampled_edges_per_hop;
  });

  return std::make_tuple(out_row, out_col, out_node_id, out_edge_id,
                         num_sampled_nodes_per_hop, num_sampled_edges_per_hop);
}

// Dispatcher //////////////////////////////////////////////////////////////////

#define DISPATCH_SAMPLE(return_edge_id, ...) \
  if (return_edge_id)                        \
    return sample<true>(__VA_ARGS__);        \
  else                                       \
    return sample<false>(__VA_ARGS__);

}  // namespace

std::tuple<at::Tensor,
           at::Tensor,
           at::Tensor,
           c10::optional<at::Tensor>,
           std::vector<int64_t>,
           std::vector<int64_t>>
labor_sample_kernel(const at::Tensor& rowptr,
                    const at::Tensor& col,
                    const at::Tensor& seed,
                    const std::vector<int64_t>& num_neighbors,
                    c10::optional<int64_t> random_seed,
                    int64_t importance_sampling,
                    bool layer_dependency,
                    bool csc,
                    bool return_edge_id) {
  TORCH_CHECK(importance_sampling == 0,
              "importance sampling is not yet supported");
  int64_t random_seed_;
  if (random_seed.has_value()) {
    random_seed_ = random_seed.value();
  } else {
    random::RandintEngine<int64_t> ng;
    random_seed_ = ng(0, 1ll << 62);
  }
  DISPATCH_SAMPLE(return_edge_id, rowptr, col, seed, num_neighbors, csc,
                  layer_dependency, random_seed_);
}

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::labor_sample"),
         TORCH_FN(labor_sample_kernel));
}

}  // namespace sampler
}  // namespace pyg
