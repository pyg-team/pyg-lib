#include "../spmm.h"

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <algorithm>
#include <vector>

namespace pyg {
namespace ops {

namespace {

at::Tensor spmm_sum_kernel(const at::Tensor& rowptr,
                           const at::Tensor& col,
                           const std::optional<at::Tensor>& optional_value,
                           const at::Tensor& mat) {
  auto rowptr_c = rowptr.contiguous();
  auto col_c = col.contiguous();
  auto mat_c = mat.contiguous();
  std::optional<at::Tensor> value_c = std::nullopt;
  if (optional_value.has_value())
    value_c = optional_value.value().contiguous();

  auto sizes = mat_c.sizes().vec();
  sizes[mat_c.dim() - 2] = rowptr_c.numel() - 1;
  auto out = at::zeros(sizes, mat_c.options());

  const int64_t M = rowptr_c.numel() - 1;
  const int64_t N = mat_c.size(-2);
  const int64_t K = mat_c.size(-1);
  int64_t B = 1;
  for (int64_t i = 0; i < mat_c.dim() - 2; ++i)
    B *= mat_c.size(i);

  if (M == 0 || K == 0 || col_c.numel() == 0)
    return out;

  const auto* rowptr_data = rowptr_c.data_ptr<int64_t>();
  const auto* col_data = col_c.data_ptr<int64_t>();
  const int64_t avg_degree = std::max<int64_t>(col_c.numel() / M, 1);
  const int64_t grain_size = std::max<int64_t>(
      1, at::internal::GRAIN_SIZE / std::max<int64_t>(K * avg_degree, 1));

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, mat_c.scalar_type(),
      "spmm_sum_cpu", [&] {
        using opmath_t = at::opmath_type<scalar_t>;

        const auto* mat_data = mat_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        const auto* value_data = value_c.has_value()
                                     ? value_c.value().data_ptr<scalar_t>()
                                     : nullptr;

        at::parallel_for(0, B * M, grain_size, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            const int64_t b = i / M;
            const int64_t m = i % M;
            const int64_t row_start = rowptr_data[m];
            const int64_t row_end = rowptr_data[m + 1];
            const int64_t mat_offset = b * N * K;
            const int64_t out_offset = b * M * K + m * K;

            if (K == 1) {
              opmath_t acc = static_cast<opmath_t>(0);
              for (int64_t e = row_start; e < row_end; ++e) {
                const int64_t c = col_data[e];
                const opmath_t x =
                    static_cast<opmath_t>(mat_data[mat_offset + c]);
                if (value_data != nullptr) {
                  acc += static_cast<opmath_t>(value_data[e]) * x;
                } else {
                  acc += x;
                }
              }
              out_data[out_offset] = static_cast<scalar_t>(acc);
            } else {
              std::vector<opmath_t> acc(K, static_cast<opmath_t>(0));
              for (int64_t e = row_start; e < row_end; ++e) {
                const int64_t c = col_data[e];
                const opmath_t v = value_data != nullptr
                                       ? static_cast<opmath_t>(value_data[e])
                                       : static_cast<opmath_t>(1);
                const int64_t src_offset = mat_offset + c * K;
                for (int64_t k = 0; k < K; ++k) {
                  acc[k] += v * static_cast<opmath_t>(mat_data[src_offset + k]);
                }
              }
              for (int64_t k = 0; k < K; ++k) {
                out_data[out_offset + k] = static_cast<scalar_t>(acc[k]);
              }
            }
          }
        });
      });

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::spmm_sum"), TORCH_FN(spmm_sum_kernel));
}

}  // namespace ops
}  // namespace pyg
