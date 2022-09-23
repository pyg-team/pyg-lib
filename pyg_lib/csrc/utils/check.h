#pragma once

#include <ATen/ATen.h>

namespace pyg {
namespace utils {

void fill_tensor_args(std::vector<at::TensorArg>& args,
                      const std::vector<at::Tensor>& tensors,
                      const std::string& name,
                      int pos);

}  // namespace utils
}  // namespace pyg
