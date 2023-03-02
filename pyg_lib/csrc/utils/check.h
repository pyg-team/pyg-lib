#pragma once

#include <ATen/ATen.h>

namespace pyg {
namespace utils {

void fill_tensor_args(std::vector<at::TensorArg>& args,
                      const at::TensorList& tensors,
                      const std::string& name,
                      int pos);

void fill_tensor_args(std::vector<at::TensorArg>& args,
                      const c10::Dict<std::string, at::Tensor>& tensors,
                      const std::string& name,
                      int pos);

}  // namespace utils
}  // namespace pyg
