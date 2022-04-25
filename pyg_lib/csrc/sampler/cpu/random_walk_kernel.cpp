#include <torch/torch.h>

#include "utils.h"

namespace pyg {
namespace sampler {

namespace {

torch::Tensor random_walk_kernel(const torch::Tensor& rowptr,
                                 const torch::Tensor& col,
                                 const torch::Tensor& seed,
                                 int64_t walk_length,
                                 double p,
                                 double q) {
  TORCH_CHECK(p == 1 && q == 1, "Uniform sampling required for now");

  const auto out = rowptr.new_empty({seed.size(0), walk_length + 1});

  AT_DISPATCH_INTEGRAL_TYPES(rowptr.scalar_type(), "random_walk_kernel", [&] {
    const auto rowptr_data = rowptr.data_ptr<scalar_t>();
    const auto col_data = col.data_ptr<scalar_t>();
    const auto seed_data = seed.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    auto grain_size = at::internal::GRAIN_SIZE / walk_length;
    at::parallel_for(0, seed.size(0), grain_size, [&](int64_t _s, int64_t _e) {
      for (auto i = _s; i < _e; i++) {
        scalar_t v = seed_data[i], row_start, row_end, rand;
        out_data[i * out.size(1) + 0] = v;  // Set seed node.

        for (auto j = 1; j < out.size(1); j++) {
          row_start = rowptr_data[v], row_end = rowptr_data[v + 1];
          if (row_end - row_start > 0)
            v = col_data[randint<scalar_t>(row_start, row_end)];
          auto rand = out_data[i * out.size(1) + j] = v;
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
