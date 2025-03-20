#include "pyg_lib/csrc/random/cpu/biased_sampling.h"

#include <ATen/Parallel.h>

namespace pyg {
namespace random {

c10::optional<at::Tensor> biased_to_cdf(const at::Tensor& rowptr,
                                        at::Tensor& bias,
                                        bool inplace) {
  TORCH_CHECK(rowptr.is_cpu(), "'rowptr' must be a CPU tensor");
  TORCH_CHECK(bias.is_cpu(), "'bias' must be a CPU tensor");

  auto cdf = at::empty_like(bias);
  // TODO: Also dispatch index type
  size_t rowptr_size = rowptr.size(0);
  int64_t* rowptr_data = rowptr.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES(bias.scalar_type(), "biased_to_cdf", [&] {
    const auto bias_data = bias.data_ptr<scalar_t>();
    scalar_t* cdf_data = nullptr;
    if (inplace) {
      cdf_data = bias.data_ptr<scalar_t>();
    } else {
      cdf_data = cdf.data_ptr<scalar_t>();
    }
    biased_to_cdf_helper(rowptr_data, rowptr_size, bias_data, cdf_data);
  });

  return cdf;
}

// The implementation of coverting to CDF representation for biased sampling.
template <typename scalar_t>
void biased_to_cdf_helper(int64_t* rowptr_data,
                          size_t rowptr_size,
                          const scalar_t* bias,
                          scalar_t* cdf) {
  at::parallel_for(0, rowptr_size - 1, at::internal::GRAIN_SIZE / rowptr_size,
                   [&](size_t _s, size_t _e) {
                     // CDF conversion for each row
                     for (size_t i = _s; i < _e; i++) {
                       const scalar_t* beg = bias + rowptr_data[i];
                       size_t len = rowptr_data[i + 1] - rowptr_data[i];
                       scalar_t* out_beg = cdf + rowptr_data[i];

                       // Remember sum and current element to
                       // enable the in-place option (bias == cdf).
                       scalar_t sum = 0;

                       for (size_t j = 0; j < len; j++) {
                         sum += beg[j];
                       }

                       scalar_t cur = sum;
                       for (size_t j = len; j > 0; j--) {
                         cur -= beg[j - 1];
                         out_beg[j - 1] = cur / sum;
                       }
                     }
                   });
}

std::pair<at::Tensor, at::Tensor> biased_to_alias(at::Tensor rowptr,
                                                  at::Tensor bias) {
  TORCH_CHECK(rowptr.is_cpu(), "'rowptr' must be a CPU tensor");
  TORCH_CHECK(bias.is_cpu(), "'bias' must be a CPU tensor");

  at::Tensor alias = at::empty_like(bias, rowptr.options());
  at::Tensor out_bias = bias.clone();

  size_t rowptr_size = rowptr.size(0);
  int64_t* rowptr_data = rowptr.data_ptr<int64_t>();
  int64_t* alias_data = alias.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES(bias.scalar_type(), "biased_to_cdf_inplace", [&] {
    scalar_t* bias_data = bias.data_ptr<scalar_t>();
    scalar_t* out_bias_data = out_bias.data_ptr<scalar_t>();
    biased_to_alias_helper(rowptr_data, rowptr_size, bias_data, out_bias_data,
                           alias_data);
  });

  return {out_bias, alias};
}

template <typename scalar_t>
void biased_to_alias_helper(int64_t* rowptr_data,
                            size_t rowptr_size,
                            const scalar_t* bias,
                            scalar_t* out_bias,
                            int64_t* alias) {
  scalar_t eps = 1e-6;
  at::parallel_for(
      0, rowptr_size - 1, at::internal::GRAIN_SIZE / rowptr_size,
      [&](size_t _s, size_t _e) {
        // Calculate the average bias
        for (size_t i = _s; i < _e; i++) {
          const scalar_t* beg = bias + rowptr_data[i];
          size_t len = rowptr_data[i + 1] - rowptr_data[i];
          scalar_t* out_beg = out_bias + rowptr_data[i];
          int64_t* alias_beg = alias + rowptr_data[i];
          scalar_t avg = 0;

          for (size_t j = 0; j < len; j++) {
            avg += beg[j];
          }
          avg /= len;

          // The sets for index with a bias lower or higher than average
          std::vector<size_t> high, low;

          low.reserve(len / 2 + 1);
          high.reserve(len / 2 + 1);

          for (size_t j = 0; j < len; j++) {
            scalar_t b = beg[j];
            // Allow some floating point error
            if (b > avg + eps) {
              high.push_back(j);
            } else if (b < avg - eps) {
              low.push_back(j);
            } else {  // if close to avg, make it a stable entry
              out_beg[j] = 1;
              alias_beg[j] = j;
            }
          }

          // Keep merging two elements, one from the lower bias set and the
          // other from the higher bias set.
          while (!low.empty()) {
            auto low_idx = low.back();

            // An index with bias lower than average means another higher one.
            TORCH_CHECK(
                !high.empty(),
                "every bias lower than avg should have a higher counterpart");
            auto high_idx = high.back();
            low.pop_back();
            high.pop_back();

            // Handle the lower one:
            auto low_bias = out_beg[low_idx];
            out_beg[low_idx] = low_bias / avg;
            alias_beg[low_idx] = high_idx;
            out_beg[high_idx] -= avg - low_bias;

            // Handle the higher one:
            scalar_t high_bias_left = out_beg[high_idx];

            // Dispatch the remaining bias to the corresponding set.
            if (high_bias_left > avg + eps) {
              high.push_back(high_idx);
            } else if (high_bias_left < avg - eps) {
              low.push_back(high_idx);
            } else {
              out_beg[high_idx] = 1;
              alias_beg[high_idx] = high_idx;
            }
          }
        }
      });
}

}  // namespace random

}  // namespace pyg
