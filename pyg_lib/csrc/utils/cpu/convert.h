#pragma once

#include <ATen/ATen.h>

namespace pyg {
namespace utils {

template <typename scalar_t>
at::Tensor from_vector(const std::vector<scalar_t>& vec, bool inplace = false) {
  int64_t size = vec.size();
  auto out = at::from_blob((scalar_t*)vec.data(), {size},
                           c10::CppTypeToScalarType<scalar_t>::value);
  return inplace ? out : out.clone();
}

template <typename scalar_t>
at::Tensor from_vector(const std::vector<std::pair<scalar_t, scalar_t>>& vec,
                       bool inplace = false) {
  int64_t size = vec.size();
  auto out = at::from_blob((scalar_t*)vec.data(), {2 * size},
                           c10::CppTypeToScalarType<scalar_t>::value);
  out = out.view({size, 2});
  return inplace ? out : out.clone();
}

template <typename scalar_t>
std::vector<scalar_t> to_vector(const at::Tensor& vec) {
  TORCH_CHECK(vec.is_contiguous(), "'vec' needs to be contiguous");
  const auto vec_data = vec.data_ptr<scalar_t>();
  return std::vector<scalar_t>(vec_data, vec_data + vec.numel());
}

}  // namespace utils
}  // namespace pyg
