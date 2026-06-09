#include "../spmm.h"

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

namespace pyg {
namespace ops {

namespace {

enum class Reduction { Sum, Mean, Min, Max };

std::tuple<at::Tensor, std::optional<at::Tensor>> spmm_reduce_kernel(
    const at::Tensor& rowptr,
    const at::Tensor& col,
    const std::optional<at::Tensor>& optional_value,
    const at::Tensor& mat,
    Reduction reduce) {
  auto rowptr_c = rowptr.contiguous();
  auto col_c = col.contiguous();
  auto mat_c = mat.contiguous();
  std::optional<at::Tensor> value_c = std::nullopt;
  if (optional_value.has_value())
    value_c = optional_value.value().contiguous();

  auto sizes = mat_c.sizes().vec();
  sizes[mat_c.dim() - 2] = rowptr_c.numel() - 1;
  auto out = at::zeros(sizes, mat_c.options());
  std::optional<at::Tensor> arg_out = std::nullopt;
  if (reduce == Reduction::Min || reduce == Reduction::Max)
    arg_out = at::full(sizes, col_c.numel(), rowptr_c.options());

  const int64_t M = rowptr_c.numel() - 1;
  const int64_t N = mat_c.size(-2);
  const int64_t K = mat_c.size(-1);
  int64_t B = 1;
  for (int64_t i = 0; i < mat_c.dim() - 2; ++i)
    B *= mat_c.size(i);

  if (M == 0 || K == 0 || col_c.numel() == 0)
    return std::make_tuple(out, arg_out);

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
        auto* arg_out_data =
            arg_out.has_value() ? arg_out.value().data_ptr<int64_t>() : nullptr;
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
              opmath_t acc;
              if (reduce == Reduction::Min) {
                acc = std::numeric_limits<opmath_t>::max();
              } else if (reduce == Reduction::Max) {
                acc = std::numeric_limits<opmath_t>::lowest();
              } else {
                acc = static_cast<opmath_t>(0);
              }
              int64_t arg = col_c.numel();
              for (int64_t e = row_start; e < row_end; ++e) {
                const int64_t c = col_data[e];
                const opmath_t x =
                    static_cast<opmath_t>(mat_data[mat_offset + c]);
                const opmath_t v = value_data != nullptr
                                       ? static_cast<opmath_t>(value_data[e])
                                       : static_cast<opmath_t>(1);
                const opmath_t val = v * x;
                if (reduce == Reduction::Min) {
                  if (val < acc) {
                    acc = val;
                    arg = e;
                  }
                } else if (reduce == Reduction::Max) {
                  if (val > acc) {
                    acc = val;
                    arg = e;
                  }
                } else {
                  acc += val;
                }
              }
              if (reduce == Reduction::Mean) {
                const int64_t count = std::max<int64_t>(row_end - row_start, 1);
                acc /= static_cast<opmath_t>(count);
              } else if (row_end == row_start && (reduce == Reduction::Min ||
                                                  reduce == Reduction::Max)) {
                acc = static_cast<opmath_t>(0);
              }
              out_data[out_offset] = static_cast<scalar_t>(acc);
              if (arg_out_data != nullptr)
                arg_out_data[out_offset] = arg;
            } else {
              std::vector<opmath_t> acc(K);
              std::vector<int64_t> args(K, col_c.numel());
              for (int64_t k = 0; k < K; ++k) {
                if (reduce == Reduction::Min) {
                  acc[k] = std::numeric_limits<opmath_t>::max();
                } else if (reduce == Reduction::Max) {
                  acc[k] = std::numeric_limits<opmath_t>::lowest();
                } else {
                  acc[k] = static_cast<opmath_t>(0);
                }
              }
              for (int64_t e = row_start; e < row_end; ++e) {
                const int64_t c = col_data[e];
                const opmath_t v = value_data != nullptr
                                       ? static_cast<opmath_t>(value_data[e])
                                       : static_cast<opmath_t>(1);
                const int64_t src_offset = mat_offset + c * K;
                for (int64_t k = 0; k < K; ++k) {
                  const opmath_t val =
                      v * static_cast<opmath_t>(mat_data[src_offset + k]);
                  if (reduce == Reduction::Min) {
                    if (val < acc[k]) {
                      acc[k] = val;
                      args[k] = e;
                    }
                  } else if (reduce == Reduction::Max) {
                    if (val > acc[k]) {
                      acc[k] = val;
                      args[k] = e;
                    }
                  } else {
                    acc[k] += val;
                  }
                }
              }
              const int64_t count = std::max<int64_t>(row_end - row_start, 1);
              for (int64_t k = 0; k < K; ++k) {
                if (reduce == Reduction::Mean) {
                  acc[k] /= static_cast<opmath_t>(count);
                } else if (row_end == row_start && (reduce == Reduction::Min ||
                                                    reduce == Reduction::Max)) {
                  acc[k] = static_cast<opmath_t>(0);
                }
                out_data[out_offset + k] = static_cast<scalar_t>(acc[k]);
                if (arg_out_data != nullptr)
                  arg_out_data[out_offset + k] = args[k];
              }
            }
          }
        });
      });

  return std::make_tuple(out, arg_out);
}

at::Tensor spmm_sum_kernel(const at::Tensor& rowptr,
                           const at::Tensor& col,
                           const std::optional<at::Tensor>& optional_value,
                           const at::Tensor& mat) {
  return std::get<0>(
      spmm_reduce_kernel(rowptr, col, optional_value, mat, Reduction::Sum));
}

at::Tensor spmm_mean_kernel(const at::Tensor& rowptr,
                            const at::Tensor& col,
                            const std::optional<at::Tensor>& optional_value,
                            const at::Tensor& mat) {
  return std::get<0>(
      spmm_reduce_kernel(rowptr, col, optional_value, mat, Reduction::Mean));
}

std::tuple<at::Tensor, at::Tensor> spmm_min_kernel(
    const at::Tensor& rowptr,
    const at::Tensor& col,
    const std::optional<at::Tensor>& optional_value,
    const at::Tensor& mat) {
  auto result =
      spmm_reduce_kernel(rowptr, col, optional_value, mat, Reduction::Min);
  return std::make_tuple(std::get<0>(result), std::get<1>(result).value());
}

std::tuple<at::Tensor, at::Tensor> spmm_max_kernel(
    const at::Tensor& rowptr,
    const at::Tensor& col,
    const std::optional<at::Tensor>& optional_value,
    const at::Tensor& mat) {
  auto result =
      spmm_reduce_kernel(rowptr, col, optional_value, mat, Reduction::Max);
  return std::make_tuple(std::get<0>(result), std::get<1>(result).value());
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::spmm_sum"), TORCH_FN(spmm_sum_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spmm_mean"), TORCH_FN(spmm_mean_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spmm_min"), TORCH_FN(spmm_min_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spmm_max"), TORCH_FN(spmm_max_kernel));
}

}  // namespace ops
}  // namespace pyg
