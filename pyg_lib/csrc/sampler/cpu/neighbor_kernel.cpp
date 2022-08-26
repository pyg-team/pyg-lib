#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch/script.h>

#include "../neighbor.h"
#include "pyg_lib/csrc/random/cpu/rand_engine.h"
#include "pyg_lib/csrc/sampler/cpu/mapper.h"
#include "pyg_lib/csrc/utils/cpu/convert.h"

#include "parallel_hashmap/phmap.h"

namespace pyg {
namespace sampler {

namespace {

template <bool replace, bool directed>
std::tuple<at::Tensor, at::Tensor, at::Tensor, c10::optional<at::Tensor>>
sample(const at::Tensor& rowptr,
       const at::Tensor& col,
       const at::Tensor& seed,
       const std::vector<int64_t>& num_neighbors) {
  AT_DISPATCH_INTEGRAL_TYPES(seed.scalar_type(), "sample_kernel", [&] {
    const auto num_nodes = rowptr.size(0) - 1;

    const auto* rowptr_data = rowptr.data_ptr<scalar_t>();
    const auto* col_data = col.data_ptr<scalar_t>();
    const auto* seed_data = seed.data_ptr<scalar_t>();

    pyg::random::RandintEngine<scalar_t> eng;

    // Initialize some data structures for the sampling process:
    std::vector<scalar_t> samples, rows, cols, edges;
    // TODO (matthias) Approximate number of sampled entries for mapper.
    auto mapper = pyg::sampler::Mapper<scalar_t>(num_nodes, seed.size(0));

    for (size_t i = 0; i < seed.numel(); i++) {
      samples.push_back(seed_data[i]);
      mapper.insert(seed_data[i]);
    }

    size_t begin = 0, end = samples.size();
    for (size_t ell = 0; ell < num_neighbors.size(); ++ell) {
      const auto num_samples = num_neighbors[ell];

      for (size_t i = begin; i < end; i++) {
        const auto v = samples[i];
        const auto row_start = rowptr_data[v];
        const auto row_end = rowptr_data[v + 1];
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
              edges.push_back(e);
            }
          }
        } else {
          std::unordered_set<scalar_t> rnd_indices;
          for (scalar_t j = row_count - num_samples; j < row_count; ++j) {
            scalar_t rnd = eng(0, j);
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
            edges.push_back(e);
          }
        }
      }
    }

    return std::make_tuple(pyg::utils::from_vector<scalar_t>(rows),
                           pyg::utils::from_vector<scalar_t>(cols),
                           pyg::utils::from_vector<scalar_t>(samples),
                           pyg::utils::from_vector<scalar_t>(edges));
  });
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
  if (disjoint) {
    AT_ERROR("Disjoint subgraphs are currently not supported");
  }
  if (!return_edge_id) {
    AT_ERROR("The indices of edges of the original graph must be returned");
  }

  if (replace && directed) {
    return sample<true, true>(rowptr, col, seed, num_neighbors);
  } else if (replace && !directed) {
    return sample<true, false>(rowptr, col, seed, num_neighbors);
  } else if (!replace && directed) {
    return sample<false, true>(rowptr, col, seed, num_neighbors);
  } else {
    return sample<false, false>(rowptr, col, seed, num_neighbors);
  }
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::neighbor_sample"),
         TORCH_FN(neighbor_sample_kernel));
}

}  // namespace sampler
}  // namespace pyg
