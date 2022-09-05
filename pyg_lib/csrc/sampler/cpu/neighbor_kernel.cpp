#include <ATen/ATen.h>
#include <torch/library.h>

#include "parallel_hashmap/phmap.h"
#include "pyg_lib/csrc/random/cpu/rand_engine.h"
#include "pyg_lib/csrc/sampler/cpu/mapper.h"
#include "pyg_lib/csrc/sampler/subgraph.h"
#include "pyg_lib/csrc/utils/cpu/convert.h"

namespace pyg {
namespace sampler {

namespace {

// `node_t` is either a scalar or a pair of scalars of (example_id, node_id):
template <typename node_t,
          typename scalar_t,
          bool replace,
          bool save_edges,
          bool save_edge_ids>
class NeighborSampler {
 public:
  NeighborSampler(const scalar_t* rowptr, const scalar_t* col)
      : rowptr_(rowptr), col_(col) {}

  void uniform_sample(const node_t& global_src_node,
                      const scalar_t& local_src_node,
                      const size_t& count,
                      pyg::sampler::Mapper<node_t, scalar_t>& dst_mapper,
                      pyg::random::RandintEngine<scalar_t>& generator,
                      std::vector<node_t>& out_global_dst_nodes) {
    if (count == 0)
      return;

    const auto offset = rowptr_offset(rowptr_, global_src_node);
    const auto row_start = std::get<0>(offset);
    const auto row_end = std::get<1>(offset);
    const auto population = row_end - row_start;

    if (population == 0)
      return;

    // Case 1: Sample the full neighborhood:
    if (count < 0 || (!replace && count >= population)) {
      for (scalar_t edge_id = row_start; edge_id < row_end; ++edge_id) {
        add(edge_id, global_src_node, local_src_node, dst_mapper,
            out_global_dst_nodes);
      }
    }

    // Case 2: Sample with replacement:
    else if (replace) {
      for (size_t i = 0; i < count; ++i) {
        const auto edge_id = generator(row_start, row_end);
        add(edge_id, global_src_node, local_src_node, dst_mapper,
            out_global_dst_nodes);
      }
    }

    // Case 3: Sample without replacement:
    else {
      std::unordered_set<scalar_t> rnd_indices;
      for (size_t i = population - count; i < population; ++i) {
        auto rnd = generator(0, i + 1);
        if (!rnd_indices.insert(rnd).second) {
          rnd = i;
          rnd_indices.insert(i);
        }
        const auto edge_id = row_start + rnd;
        add(edge_id, global_src_node, local_src_node, dst_mapper,
            out_global_dst_nodes);
      }
    }
  }

  std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
  get_sampled_edges() {
    TORCH_CHECK(save_edges, "No edges have been stored")
    const auto row = pyg::utils::from_vector(sampled_rows_);
    const auto col = pyg::utils::from_vector(sampled_cols_);
    c10::optional<at::Tensor> edge_id = c10::nullopt;
    if (save_edge_ids) {
      edge_id = pyg::utils::from_vector(sampled_edge_ids_);
    }
    return std::make_tuple(row, col, edge_id);
  }

 private:
  inline std::pair<scalar_t, scalar_t> rowptr_offset(const scalar_t* rowptr,
                                                     const scalar_t& node) {
    return {rowptr[node], rowptr[node + 1]};
  }

  inline std::pair<scalar_t, scalar_t> rowptr_offset(
      const scalar_t* rowptr,
      const std::pair<scalar_t, scalar_t>& node) {
    return {rowptr[std::get<1>(node)], rowptr[std::get<1>(node) + 1]};
  }

  inline scalar_t to_node_t(const scalar_t& node, const scalar_t& ref) {
    return node;
  }

  inline std::pair<scalar_t, scalar_t> to_node_t(
      const scalar_t& node,
      const std::pair<scalar_t, scalar_t>& ref) {
    return {std::get<0>(ref), node};
  }

  inline void add(const scalar_t& edge_id,
                  const node_t& global_src_node,
                  const scalar_t& local_src_node,
                  pyg::sampler::Mapper<node_t, scalar_t>& dst_mapper,
                  std::vector<node_t>& out_global_dst_nodes) {
    const auto global_dst_node_value = col_[edge_id];
    const auto global_dst_node =
        to_node_t(global_dst_node_value, global_src_node);
    const auto res = dst_mapper.insert(global_dst_node);
    if (res.second) {  // not yet sampled.
      out_global_dst_nodes.push_back(global_dst_node);
    }
    if (save_edges) {
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
};

template <bool replace, bool directed, bool disjoint, bool return_edge_id>
std::tuple<at::Tensor, at::Tensor, at::Tensor, c10::optional<at::Tensor>>
sample(const at::Tensor& rowptr,
       const at::Tensor& col,
       const at::Tensor& seed,
       const std::vector<int64_t>& num_neighbors) {
  at::Tensor out_row, out_col, out_node_id;
  c10::optional<at::Tensor> out_edge_id = c10::nullopt;

  AT_DISPATCH_INTEGRAL_TYPES(seed.scalar_type(), "sample_kernel", [&] {
    typedef std::pair<scalar_t, scalar_t> pair_scalar_t;
    typedef std::conditional_t<!disjoint, scalar_t, pair_scalar_t> node_t;

    pyg::random::RandintEngine<scalar_t> generator;

    auto mapper = Mapper<node_t, scalar_t>(/*num_nodes=*/rowptr.size(0) - 1);
    std::vector<node_t> sampled_nodes;

    const auto seed_data = seed.data_ptr<scalar_t>();
    for (size_t i = 0; i < seed.numel(); i++) {
      if constexpr (!disjoint) {
        mapper.insert(seed_data[i]);
        sampled_nodes.push_back(seed_data[i]);
      } else {
        mapper.insert({i, seed_data[i]});
        sampled_nodes.push_back({i, seed_data[i]});
      }
    }

    auto sampler =
        NeighborSampler<node_t, scalar_t, replace, directed, return_edge_id>(
            rowptr.data_ptr<scalar_t>(), col.data_ptr<scalar_t>());

    size_t begin = 0, end = seed.size(0);
    for (size_t ell = 0; ell < num_neighbors.size(); ++ell) {
      const auto count = num_neighbors[ell];

      for (size_t i = begin; i < end; ++i) {
        sampler.uniform_sample(/*global_src_node=*/sampled_nodes[i],
                               /*local_src_node=*/i, count, mapper, generator,
                               /*out_global_dst_nodes=*/sampled_nodes);
      }
      begin = end, end = sampled_nodes.size();
    }

    if constexpr (!disjoint)
      out_node_id = pyg::utils::from_vector(sampled_nodes);
    else {
      std::vector<scalar_t> sampled_node_values(sampled_nodes.size());
      for (const node_t& v : sampled_nodes)
        sampled_node_values.push_back(v.second);
      out_node_id = pyg::utils::from_vector(sampled_node_values);
    }

    if (directed) {
      std::tie(out_row, out_col, out_edge_id) = sampler.get_sampled_edges();
    } else {
      TORCH_CHECK(!disjoint, "Disjoint subgraphs are not yet supported");
      std::tie(out_row, out_col, out_edge_id) =
          pyg::sampler::subgraph(rowptr, col, out_node_id, return_edge_id);
    }
  });
  return std::make_tuple(out_row, out_col, out_node_id, out_edge_id);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, c10::optional<at::Tensor>>
neighbor_sample_kernel(const at::Tensor& rowptr,
                       const at::Tensor& col,
                       const at::Tensor& seed,
                       const std::vector<int64_t>& num_neighbors,
                       bool replace,
                       bool directed,
                       bool disjoint,
                       bool return_edge_id) {
  if (disjoint)
    return sample<false, true, true, true>(rowptr, col, seed, num_neighbors);
  else
    return sample<false, true, false, true>(rowptr, col, seed, num_neighbors);

  /* if (return_edge_id) { */
  /*   if (replace && directed) { */
  /*     return sample<true, true, true>(rowptr, col, seed, num_neighbors);
   */
  /*   } else if (replace && !directed) { */
  /*     return sample<true, false, true>(rowptr, col, seed, num_neighbors);
   */
  /*   } else if (!replace && directed) { */
  /*     return sample<false, true, true>(rowptr, col, seed, num_neighbors);
   */
  /*   } else { */
  /*     return sample<false, false, true>(rowptr, col, seed,
   * num_neighbors);
   */
  /*   } */
  /* } else { */
  /*   if (replace && directed) { */
  /*     return sample<true, true, false>(rowptr, col, seed, num_neighbors);
   */
  /*   } else if (replace && !directed) { */
  /*     return sample<true, false, false>(rowptr, col, seed,
   * num_neighbors);
   */
  /*   } else if (!replace && directed) { */
  /*     return sample<false, true, false>(rowptr, col, seed,
   * num_neighbors);
   */
  /*   } else { */
  /*     return sample<false, false, false>(rowptr, col, seed,
   * num_neighbors);
   */
  /*   } */
  /* } */
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::neighbor_sample"),
         TORCH_FN(neighbor_sample_kernel));
}

}  // namespace sampler
}  // namespace pyg
