#include "pyg_lib/csrc/random/cpu/biased_sampling.h"

namespace pyg {
namespace random {

at::Tensor biased_to_cdf(at::Tensor rowptr, at::Tensor bias) {
  TORCH_CHECK(rowptr.is_cpu(), "'rowptr' must be a CPU tensor");
  TORCH_CHECK(bias.is_cpu(), "'bias' must be a CPU tensor");

  auto cdf = at::empty_like(bias);
  int64_t rowptr_size = rowptr.size(0);
  int64_t* rowptr_data = rowptr.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES(bias.scalar_type(), "biased_to_cdf", [&] {
    scalar_t* bias_data = bias.data_ptr<scalar_t>();
    scalar_t* cdf_data = cdf.data_ptr<scalar_t>();
    biased_to_cdf_helper(rowptr_data, rowptr_size, bias_data, cdf_data);
  });

  return cdf;
}

void biased_to_cdf_inplace(at::Tensor rowptr, at::Tensor bias) {
  TORCH_CHECK(rowptr.is_cpu(), "'rowptr' must be a CPU tensor");
  TORCH_CHECK(bias.is_cpu(), "'bias' must be a CPU tensor");

  int64_t rowptr_size = rowptr.size(0);
  int64_t* rowptr_data = rowptr.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES(bias.scalar_type(), "biased_to_cdf_inplace", [&] {
    scalar_t* bias_data = bias.data_ptr<scalar_t>();
    biased_to_cdf_helper(rowptr_data, rowptr_size, bias_data, bias_data);
  });
}
}  // namespace random

}  // namespace pyg
