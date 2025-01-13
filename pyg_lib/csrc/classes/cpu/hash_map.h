#pragma once

#include <ATen/ATen.h>
#include "parallel_hashmap/phmap.h"

namespace pyg {
namespace classes {

struct IHashMap {
  virtual ~IHashMap() = default;
  virtual at::Tensor get(const at::Tensor& query) = 0;
};

template <typename KeyType>
struct CPUHashMapImpl : IHashMap {
 public:
  using ValueType = int64_t;

  CPUHashMapImpl(const at::Tensor& key);
  at::Tensor get(const at::Tensor& query) override;

 private:
  phmap::parallel_flat_hash_map<
      KeyType,
      ValueType,
      phmap::priv::hash_default_hash<KeyType>,
      phmap::priv::hash_default_eq<KeyType>,
      phmap::priv::Allocator<std::pair<const KeyType, ValueType>>,
      12,
      std::mutex>
      map_;
};

struct CPUHashMap : torch::CustomClassHolder {
 public:
  CPUHashMap(const at::Tensor& key);
  at::Tensor get(const at::Tensor& query);

 private:
  std::unique_ptr<IHashMap> map_;
};

}  // namespace classes
}  // namespace pyg
