#pragma once

#include <ATen/ATen.h>
#include "parallel_hashmap/phmap.h"

namespace pyg {
namespace classes {

template <typename KeyType>
struct CPUHashMap : torch::CustomClassHolder {
 public:
  using ValueType = int64_t;

  CPUHashMap(const at::Tensor& key);
  at::Tensor get(const at::Tensor& query);

 private:
  phmap::parallel_flat_hash_map<
      KeyType,
      ValueType,
      phmap::priv::hash_default_hash<KeyType>,
      phmap::priv::hash_default_eq<KeyType>,
      phmap::priv::Allocator<std::pair<const KeyType, ValueType>>,
      8,
      phmap::NullMutex>
      map_;
};

}  // namespace classes
}  // namespace pyg
