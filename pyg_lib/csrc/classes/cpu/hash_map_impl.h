#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <parallel_hashmap/phmap.h>

#include "../hash_map_impl.h"

namespace pyg {
namespace classes {

template <typename KeyType>
struct CPUHashMapImpl : HashMapImpl {
 public:
  using ValueType = int64_t;

  CPUHashMapImpl(const at::Tensor& key) {
    map_.reserve(key.numel());

    const auto num_threads = at::get_num_threads();
    const auto grain_size =
        std::max((key.numel() + num_threads - 1) / num_threads,
                 at::internal::GRAIN_SIZE);
    const auto key_data = key.data_ptr<KeyType>();

    at::parallel_for(0, key.numel(), grain_size, [&](int64_t beg, int64_t end) {
      for (int64_t i = beg; i < end; ++i) {
        auto [iterator, inserted] = map_.insert({key_data[i], i});
        TORCH_CHECK(inserted, "Found duplicated key in 'HashMap'.");
      }
    });
  }

  at::Tensor get(const at::Tensor& query) override {
    const auto options = at::TensorOptions().dtype(at::kLong);
    const auto out = at::empty({query.numel()}, options);
    auto out_data = out.data_ptr<int64_t>();

    const auto num_threads = at::get_num_threads();
    const auto grain_size =
        std::max((query.numel() + num_threads - 1) / num_threads,
                 at::internal::GRAIN_SIZE);
    const auto query_data = query.data_ptr<int64_t>();

    at::parallel_for(0, query.numel(), grain_size, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i) {
        auto it = map_.find(query_data[i]);
        out_data[i] = (it != map_.end()) ? it->second : -1;
      }
    });

    return out;
  }

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

}  // namespace classes
}  // namespace pyg
