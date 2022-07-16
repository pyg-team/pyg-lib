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

inline at::IntArrayRef sizes_from_ptr(const at::Tensor& ptr) {
  auto size = ptr.narrow(/*dim=*/0, /*start=*/1, /*length=*/ptr.numel() - 1) -
              ptr.narrow(/*dim=*/0, /*start=*/0, /*length=*/ptr.numel() - 1);

  size = size.cpu();  // For now, we need to transfer from device to host.

  // TODO (matthias) Allow for other types than `int64_t`.
  return at::IntArrayRef(size.data_ptr<int64_t>(), size.numel());
}

}  // namespace utils
}  // namespace pyg
