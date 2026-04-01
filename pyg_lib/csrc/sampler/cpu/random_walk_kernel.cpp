#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "pyg_lib/csrc/random/cpu/rand_engine.h"

namespace pyg {
namespace sampler {

namespace {

std::tuple<at::Tensor, at::Tensor> random_walk_kernel(const at::Tensor& rowptr,
                                                      const at::Tensor& col,
                                                      const at::Tensor& seed,
                                                      int64_t walk_length,
                                                      double p,
                                                      double q) {
  TORCH_CHECK(rowptr.is_cpu(), "'rowptr' must be a CPU tensor");
  TORCH_CHECK(col.is_cpu(), "'col' must be a CPU tensor");
  TORCH_CHECK(seed.is_cpu(), "'seed' must be a CPU tensor");
  TORCH_CHECK(p == 1 && q == 1, "Uniform sampling required for now");

  const auto node_seq = rowptr.new_empty({seed.size(0), walk_length + 1});
  const auto edge_seq = rowptr.new_full({seed.size(0), walk_length}, -1);

  AT_DISPATCH_INTEGRAL_TYPES(seed.scalar_type(), "random_walk_kernel", [&] {
    const auto rowptr_data = rowptr.data_ptr<scalar_t>();
    const auto col_data = col.data_ptr<scalar_t>();
    const auto seed_data = seed.data_ptr<scalar_t>();
    auto node_data = node_seq.data_ptr<scalar_t>();
    auto edge_data = edge_seq.data_ptr<scalar_t>();

    auto grain_size = at::internal::GRAIN_SIZE / walk_length;
    at::parallel_for(0, seed.size(0), grain_size, [&](int64_t _s, int64_t _e) {
      pyg::random::RandintEngine<scalar_t> eng;
      for (auto i = _s; i < _e; ++i) {
        auto v = seed_data[i];
        node_data[i * (walk_length + 1) + 0] = v;  // Set seed node.

        for (auto j = 0; j < walk_length; ++j) {
          auto row_start = rowptr_data[v], row_end = rowptr_data[v + 1];
          if (row_end - row_start > 0) {
            auto edge_idx = eng(row_start, row_end);
            v = col_data[edge_idx];
            edge_data[i * walk_length + j] = edge_idx;
          }
          // For isolated nodes, this will add a fake self-loop.
          // This does not do any harm when used in within a `node2vec` model.
          // edge_seq remains -1 for isolated nodes.
          node_data[i * (walk_length + 1) + (j + 1)] = v;
        }
      }
    });
  });

  return std::make_tuple(node_seq, edge_seq);
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::random_walk"),
         TORCH_FN(random_walk_kernel));
}

}  // namespace sampler
}  // namespace pyg
