#include <tuple>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "radix_sort.h"

namespace pyg {
namespace ops {

namespace {

template <typename scalar_t>
void vectorized_copy(scalar_t* dst, const scalar_t* src, int64_t size) {
  constexpr int64_t unfold_step = 4;
  int64_t index;
  if (size >= unfold_step) {
#pragma omp simd
    for (index = 0; index < size; index += unfold_step) {
      dst[index] = src[index];
      dst[index + 1] = src[index + 1];
      dst[index + 2] = src[index + 2];
      dst[index + 3] = src[index + 3];
    }
  }
  for (index; index < size; ++index) {
    dst[index] = src[index];
  }
}

std::tuple<at::Tensor, at::Tensor> index_sort_kernel(
    const at::Tensor& input,
    const at::optional<int64_t> max) {
  TORCH_CHECK(input.is_contiguous(), "Input should be contiguous.")
  TORCH_CHECK(input.dim() == 1, "Input should be 1-dimensional.");
  if (input.numel() > at::internal::GRAIN_SIZE && is_radix_sort_available()) {
    const auto elements = input.numel();
    const auto maximum = max.value_or(at::max(input).item<int64_t>());
    auto out_vals = at::detach(input).clone();
    auto out_indices = at::arange(
        0, elements, at::TensorOptions().device(at::kCPU).dtype(at::kLong));

    AT_DISPATCH_INTEGRAL_TYPES(
        out_vals.scalar_type(), "index_sort_kernel", [&] {
          scalar_t* vals = out_vals.data_ptr<scalar_t>();
          int64_t* indices = out_indices.data_ptr<int64_t>();
          std::vector<scalar_t> tmp_vals(elements);
          std::vector<int64_t> tmp_indices(elements);
          scalar_t* sorted_vals = nullptr;
          int64_t* sorted_indices = nullptr;
          std::tie(sorted_vals, sorted_indices) =
              radix_sort_parallel(vals, indices, tmp_vals.data(),
                                  tmp_indices.data(), elements, maximum);

          const bool sorted_in_place = vals == sorted_vals;
          if (!sorted_in_place) {
            const int num_threads = at::get_num_threads();
            const auto common_size = out_vals.numel();
            at::parallel_for(
                0, common_size, at::internal::GRAIN_SIZE / num_threads,
                [&](int64_t begin, int64_t end) {
                  const auto job_size = end - begin;
                  vectorized_copy(vals + begin, sorted_vals + begin, job_size);
                  vectorized_copy(indices + begin, sorted_indices + begin,
                                  job_size);
                });
          }
        });
    return std::tuple<at::Tensor, at::Tensor>(out_vals, out_indices);
  } else {
    TORCH_CHECK(at::isIntegralType(input.scalar_type(), /*includeBool=*/false),
                "Input should contain integral values.");
    return at::sort(input);
  }
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::index_sort"), TORCH_FN(index_sort_kernel));
}

}  // namespace ops
}  // namespace pyg
