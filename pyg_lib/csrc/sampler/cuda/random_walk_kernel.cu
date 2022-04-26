#include <torch/torch.h>

namespace pyg {
namespace sampler {

namespace {

torch::Tensor random_walk_kernel(const torch::Tensor& rowptr,
                                 const torch::Tensor& col,
                                 const torch::Tensor& seed,
                                 int64_t walk_length,
                                 double p,
                                 double q) {
  TORCH_CHECK(rowptr.is_cuda(), "'rowptr' must be a CUDA tensor");
  TORCH_CHECK(col.is_cuda(), "'col' must be a CUDA tensor");
  TORCH_CHECK(seed.is_cuda(), "'seed' must be a CUDA tensor");
  TORCH_CHECK(p == 1 && q == 1, "Uniform sampling required for now");

  const auto out = rowptr.new_empty({seed.size(0), walk_length + 1});

  AT_DISPATCH_INTEGRAL_TYPES(seed.scalar_type(), "random_walk_kernel", [&] {
    const auto rowptr_data = rowptr.data_ptr<scalar_t>();
    const auto col_data = col.data_ptr<scalar_t>();
    const auto seed_data = seed.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();
  });

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::random_walk"),
         TORCH_FN(random_walk_kernel));
}

}  // namespace sampler
}  // namespace pyg
