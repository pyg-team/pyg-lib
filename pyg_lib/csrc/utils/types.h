#pragma once

#include <string>

#include <ATen/ATen.h>

namespace pyg {
namespace utils {
using RELATION_TYPE = std::string;

using HETERO_TENSOR_TYPE = c10::Dict<RELATION_TYPE, at::Tensor>;
}  // namespace utils

}  // namespace pyg
