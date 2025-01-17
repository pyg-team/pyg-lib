#pragma once

#include <ATen/ATen.h>
#include "../hash_map_impl.h"

namespace pyg {
namespace classes {

template <typename KeyType>
struct CUDAHashMapImpl : HashMapImpl {
 public:
  CUDAHashMapImpl(const at::Tensor& key) {}
  at::Tensor get(const at::Tensor& query) override { return at::empty(0); }
};

}  // namespace classes
}  // namespace pyg
