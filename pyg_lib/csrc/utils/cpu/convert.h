#pragma once

#include <ATen/ATen.h>

namespace pyg {
namespace utils {

template <typename scalar_t>
at::Tensor from_vector(const std::vector<scalar_t>& vec, bool inplace = false) {
  const auto out = at::from_blob((scalar_t*)vec.data(), {(int64_t)vec.size()},
                                 c10::CppTypeToScalarType<scalar_t>::value);
  return inplace ? out : out.clone();
}

}  // namespace utils
}  // namespace pyg
