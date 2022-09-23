#include "check.h"

namespace pyg {
namespace utils {

void fill_tensor_args(std::vector<at::TensorArg>& args,
                      const std::vector<at::Tensor>& tensors,
                      const std::string& name,
                      int pos) {
  args.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto full_name = name + "[" + std::to_string(i) + "]";
    args.emplace_back(tensors[i], full_name.c_str(), pos);
  }
}

}  // namespace utils
}  // namespace pyg
