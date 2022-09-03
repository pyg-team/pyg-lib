#include <ATen/ATen.h>
#include <torch/library.h>

#include "parallel_hashmap/phmap.h"

#include "pyg_lib/csrc/random/cpu/rand_engine.h"
#include "pyg_lib/csrc/sampler/cpu/mapper.h"
#include "pyg_lib/csrc/utils/cpu/convert.h"

namespace pyg {
namespace sampler {

namespace {

template <typename scalar_t, bool replace, bool save_edges, bool save_edge_ids>
class NeighborSampler {
 public:
  NeighborSampler(const scalar_t* rowptr, const scalar_t* col)
      : rowptr(rowptr), col(col) {}

  void uniform_sample(const scalar_t global_src_node,
                      const scalar_t local_src_node,
                      const size_t count,
                      pyg::sampler::Mapper<scalar_t>& dst_mapper,
                      pyg::random::RandintEngine<scalar_t>& generator,
                      std::vector<scalar_t>& out_global_dst_nodes) {
    if (count == 0)
      return;

    const auto row_start = rowptr[global_src_node];
    const auto row_end = rowptr[global_src_node + 1];
    const auto population = row_end - row_start;

    if (population == 0)
      return;

    // Case 1: Sample the full neighborhood:
    if (count < 0 || (!replace && count >= population)) {
      for (scalar_t edge_id = row_start; edge_id < row_end; ++edge_id) {
        add(edge_id, local_src_node, dst_mapper, out_global_dst_nodes);
      }
    }

    // Case 2: Sample with replacement:
    else if (replace) {
      for (size_t i = 0; i < count; ++i) {
        const auto edge_id = generator(row_start, row_end);
        add(edge_id, local_src_node, dst_mapper, out_global_dst_nodes);
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
        add(edge_id, local_src_node, dst_mapper, out_global_dst_nodes);
      }
    }
  }

 private:
  inline void add(const scalar_t edge_id,
                  const scalar_t local_src_node,
                  pyg::sampler::Mapper<scalar_t>& dst_mapper,
                  std::vector<scalar_t>& out_global_dst_nodes) {
    const auto global_dst_node = col[edge_id];
    const auto res = dst_mapper.insert(global_dst_node);
    if (res.second) {  // not yet sampled.
      out_global_dst_nodes.push_back(global_dst_node);
      if (save_edges) {
        sampled_rows.push_back(local_src_node);
        sampled_cols.push_back(res.first);
      }
      if (save_edge_ids) {
        sampled_edge_ids.push_back(edge_id);
      }
    }
  }

  const scalar_t* rowptr;
  const scalar_t* col;
  std::vector<scalar_t> sampled_rows;
  std::vector<scalar_t> sampled_cols;
  std::vector<scalar_t> sampled_edge_ids;
};

template <bool replace, bool directed, bool return_edge_id>
std::tuple<at::Tensor, at::Tensor, at::Tensor, c10::optional<at::Tensor>>
sample(const at::Tensor& rowptr,
       const at::Tensor& col,
       const at::Tensor& seed,
       const std::vector<int64_t>& num_neighbors) {
  at::Tensor out_row, out_col, out_node_id;
  c10::optional<at::Tensor> out_edge_id = c10::nullopt;

  AT_DISPATCH_INTEGRAL_TYPES(seed.scalar_type(), "sample_kernel", [&] {
    const auto num_nodes = rowptr.size(0) - 1;

    const auto* rowptr_data = rowptr.data_ptr<scalar_t>();
    const auto* col_data = col.data_ptr<scalar_t>();
    const auto* seed_data = seed.data_ptr<scalar_t>();

    pyg::random::RandintEngine<scalar_t> eng;

    // TODO (matthias) Approximate number of sampled entries for mapper.
    auto mapper = pyg::sampler::Mapper<scalar_t>(num_nodes, seed.size(0));
    mapper.fill(seed);

    // Initialize some data structures for the sampling process:
    std::vector<scalar_t> rows, cols, samples, edges;

    for (size_t i = 0; i < seed.numel(); i++)
      samples.push_back(seed_data[i]);

    size_t begin = 0, end = samples.size();
    for (size_t ell = 0; ell < num_neighbors.size(); ++ell) {
      const auto num_samples = num_neighbors[ell];

      for (size_t i = begin; i < end; i++) {
        const auto v = samples[i];
        const auto row_start = rowptr_data[v], row_end = rowptr_data[v + 1];
        const auto row_count = row_end - row_start;

        if (row_count == 0)
          continue;

        if ((num_samples < 0) || (!replace && (num_samples >= row_count))) {
          for (scalar_t e = row_start; e < row_end; ++e) {
            const auto w = col_data[e];
            const auto res = mapper.insert(w);
            if (res.second)
              samples.push_back(w);
            if (directed) {
              rows.push_back(i);
              cols.push_back(res.first);
              if (return_edge_id)
                edges.push_back(e);
            }
          }
        } else if (replace) {
          for (size_t j = 0; j < num_samples; ++j) {
            const scalar_t e = eng(row_start, row_end);
            const auto w = col_data[e];
            const auto res = mapper.insert(w);
            if (res.second)
              samples.push_back(w);
            if (directed) {
              rows.push_back(i);
              cols.push_back(res.first);
              if (return_edge_id)
                edges.push_back(e);
            }
          }
        } else {
          std::unordered_set<scalar_t> rnd_indices;
          for (scalar_t j = row_count - num_samples; j < row_count; ++j) {
            scalar_t rnd = eng(0, j + 1);
            if (!rnd_indices.insert(rnd).second) {
              rnd = j;
              rnd_indices.insert(j);
            }
            const scalar_t e = row_start + rnd;
            const auto w = col_data[e];
            const auto res = mapper.insert(w);
            if (res.second)
              samples.push_back(w);
            if (directed) {
              rows.push_back(i);
              cols.push_back(res.first);
              if (return_edge_id)
                edges.push_back(e);
            }
          }
        }
      }
      begin = end, end = samples.size();
    }

    if (!directed) {
      // TODO (matthias) Use pyg::sampler::subgraph() for this.
      for (size_t i = 0; i < samples.size(); ++i) {
        const auto v = samples[i];
        const auto row_start = rowptr_data[v];
        const auto row_end = rowptr_data[v + 1];
        for (scalar_t e = row_start; e < row_end; ++e) {
          const auto local_node = mapper.map(col_data[v]);
          if (local_node != -1) {
            rows.push_back(i);
            cols.push_back(local_node);
            if (return_edge_id)
              edges.push_back(e);
          }
        }
      }
    }

    out_row = pyg::utils::from_vector(rows);
    out_col = pyg::utils::from_vector(cols);
    out_node_id = pyg::utils::from_vector(samples);
    if (return_edge_id)
      out_edge_id = pyg::utils::from_vector(edges);
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
    AT_ERROR("Disjoint subgraphs are currently not supported");

  if (return_edge_id) {
    if (replace && directed) {
      return sample<true, true, true>(rowptr, col, seed, num_neighbors);
    } else if (replace && !directed) {
      return sample<true, false, true>(rowptr, col, seed, num_neighbors);
    } else if (!replace && directed) {
      return sample<false, true, true>(rowptr, col, seed, num_neighbors);
    } else {
      return sample<false, false, true>(rowptr, col, seed, num_neighbors);
    }
  } else {
    if (replace && directed) {
      return sample<true, true, false>(rowptr, col, seed, num_neighbors);
    } else if (replace && !directed) {
      return sample<true, false, false>(rowptr, col, seed, num_neighbors);
    } else if (!replace && directed) {
      return sample<false, true, false>(rowptr, col, seed, num_neighbors);
    } else {
      return sample<false, false, false>(rowptr, col, seed, num_neighbors);
    }
  }
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::neighbor_sample"),
         TORCH_FN(neighbor_sample_kernel));
}

}  // namespace sampler
}  // namespace pyg
