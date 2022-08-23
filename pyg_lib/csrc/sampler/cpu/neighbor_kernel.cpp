#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch/script.h>

#include "pyg_lib/csrc/sampler/cpu/mapper.h"
#include "utils.h"
#include "../neighbor.h"

#include "parallel_hashmap/phmap.h"

#ifdef _WIN32
#include <process.h>
#endif

namespace pyg {
namespace sampler {

namespace {

template <bool replace, bool directed>
std::tuple<at::Tensor, at::Tensor, at::Tensor, c10::optional<at::Tensor>>
sample(const at::Tensor& rowptr, const at::Tensor& col,
       const at::Tensor& seed, const std::vector<int64_t>& num_neighbors) {

   // Initialize some data structures for the sampling process:
  std::vector<int64_t> samples, rows, cols, edges;
  phmap::flat_hash_map<int64_t, int64_t> to_local_node;

  const auto num_nodes = rowptr.size(0) - 1;
  AT_DISPATCH_INTEGRAL_TYPES(seed.scalar_type(), "neighbor_kernel", [&] {
    auto *rowptr_data = rowptr.data_ptr<scalar_t>();
    auto *col_data = col.data_ptr<scalar_t>();
    auto *seed_data = seed.data_ptr<scalar_t>();

    for (int64_t i = 0; i < seed.numel(); i++) {
      const auto &v = seed_data[i];
      samples.push_back(v);
      to_local_node.insert({v, i});
    }

    int64_t begin = 0, end = samples.size();
    for (int64_t ell = 0; ell < static_cast<int64_t>(num_neighbors.size()); ell++) {

      const auto &num_samples = num_neighbors[ell];
      for (int64_t i = begin; i < end; i++) {
        const auto &w = samples[i];
        const auto &row_start = rowptr_data[w];
        const auto &row_end = rowptr_data[w + 1];
        const auto row_count = row_end - row_start;

        if (row_count == 0)
          continue;

        if ((num_samples < 0) || (!replace && (num_samples >= row_count))) {
          for (int64_t offset = row_start; offset < row_end; offset++) {
            const int64_t &v = col_data[offset];
            const auto res = to_local_node.insert({v, samples.size()});
            if (res.second)
              samples.push_back(v);
            if (directed) {
              rows.push_back(i);
              cols.push_back(res.first->second);
              edges.push_back(offset);
            }
          }
        } else if (replace) {
          for (int64_t j = 0; j < num_samples; j++) {
            const int64_t offset = row_start + uniform_randint(row_count);
            const int64_t &v = col_data[offset];
            const auto res = to_local_node.insert({v, samples.size()});
            if (res.second)
              samples.push_back(v);
            if (directed) {
              cols.push_back(i);
              rows.push_back(res.first->second);
              edges.push_back(offset);
            }
          }
        } else {
          std::unordered_set<int64_t> rnd_indices;
          for (int64_t j = row_count - num_samples; j < row_count; j++) {
            int64_t rnd = uniform_randint(j);
            if (!rnd_indices.insert(rnd).second) {
              rnd = j;
              rnd_indices.insert(j);
            }
            const int64_t offset = row_start + rnd;
            const int64_t &v = col_data[offset];
            const auto res = to_local_node.insert({v, samples.size()});
            if (res.second)
              samples.push_back(v);
            if (directed) {
              rows.push_back(i);
              cols.push_back(res.first->second);
              edges.push_back(offset);
            }
          }
        }
      }
      begin = end, end = samples.size();
    }

    if (!directed) {
      phmap::flat_hash_map<int64_t, int64_t>::iterator iter;
      for (int64_t i = 0; i < static_cast<int64_t>(samples.size()); i++) {
        const auto &w = samples[i];
        const auto &row_start = rowptr_data[w];
        const auto &row_end = rowptr_data[w + 1];
        for (int64_t offset = row_start; offset < row_end; offset++) {
          const auto &v = col_data[offset];
          iter = to_local_node.find(v);
          if (iter != to_local_node.end()) {
            cols.push_back(iter->second);
            rows.push_back(i);
            edges.push_back(offset);
          }
        }
      }
    }
  });
  return std::make_tuple(from_vector<int64_t>(rows), from_vector<int64_t>(cols),
                    from_vector<int64_t>(samples), from_vector<int64_t>(edges));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, c10::optional<at::Tensor>>
neighbor_sample_kernel(const at::Tensor& rowptr,
                       const at::Tensor& col,
                       const at::Tensor& seed,
                       const std::vector<int64_t>& num_neighbors,
                       bool replace,
                       bool directed,
                       bool isolated,
                       bool return_edge_id) {
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
