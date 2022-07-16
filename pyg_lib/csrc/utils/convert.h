#pragma once

#include <ATen/ATen.h>

namespace pyg {
namespace utils {

at::Tensor size_from_ptr(const at::Tensor& ptr);

}  // namespace utils
}  // namespace pyg
