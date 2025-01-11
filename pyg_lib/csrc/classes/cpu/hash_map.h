#pragma once

#include <ATen/ATen.h>
#include <variant>

namespace pyg {
namespace classes {

struct CPUHashMap : torch::CustomClassHolder {
 public:
  using KeyType = std::
      variant<bool, uint8_t, int8_t, int16_t, int32_t, int64_t, float, double>;

  CPUHashMap(const at::Tensor& key);
  at::Tensor get(const at::Tensor& query);

 private:
  std::unordered_map<KeyType, int64_t> map_;
};

}  // namespace classes
}  // namespace pyg
