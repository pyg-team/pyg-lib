#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "pyg_lib/csrc/random/cpu/rand_engine.h"

namespace pyg {
namespace sampler {

namespace {

at::Tensor random_walk_kernel(const at::Tensor& rowptr,
                              const at::Tensor& col,
                              const at::Tensor& seed,
                              int64_t walk_length,
                              double p,
                              double q) {
  TORCH_CHECK(rowptr.is_cpu(), "'rowptr' must be a CPU tensor");
  TORCH_CHECK(col.is_cpu(), "'col' must be a CPU tensor");
  TORCH_CHECK(seed.is_cpu(), "'seed' must be a CPU tensor");
  TORCH_CHECK(p == 1 && q == 1, "Uniform sampling required for now");

  const auto out = rowptr.new_empty({seed.size(0), walk_length + 1});

  AT_DISPATCH_INTEGRAL_TYPES(seed.scalar_type(), "random_walk_kernel", [&] {
    const auto rowptr_data = rowptr.data_ptr<scalar_t>();
    const auto col_data = col.data_ptr<scalar_t>();
    const auto seed_data = seed.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    auto grain_size = at::internal::GRAIN_SIZE / walk_length;
    at::parallel_for(0, seed.size(0), grain_size, [&](int64_t _s, int64_t _e) {
      pyg::random::RandintEngine<scalar_t> eng;
      for (auto i = _s; i < _e; ++i) {
        auto v = seed_data[i];
        out_data[i * (walk_length + 1) + 0] = v;  // Set seed node.

        for (auto j = 1; j < walk_length + 1; ++j) {
          auto row_start = rowptr_data[v], row_end = rowptr_data[v + 1];
          if (row_end - row_start > 0)
            v = col_data[eng(row_start, row_end)];
          // For isolated nodes, this will add a fake self-loop.
          // This does not do any harm when used in within a `node2vec` model.
          out_data[i * (walk_length + 1) + j] = v;
        }
      }
    });
  });

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::random_walk"),
         TORCH_FN(random_walk_kernel));
}

}  // namespace sampler
}  // namespace pyg
