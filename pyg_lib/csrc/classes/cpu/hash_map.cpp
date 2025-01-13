#include "hash_map.h"

#include <ATen/Parallel.h>
#include <torch/library.h>

namespace pyg {
namespace classes {

template <typename KeyType>
CPUHashMap<KeyType>::CPUHashMap(const at::Tensor& key) {
  at::TensorArg key_arg{key, "key", 0};
  at::CheckedFrom c{"HashMap.init"};
  at::checkDeviceType(c, key, at::DeviceType::CPU);
  at::checkDim(c, key_arg, 1);
  at::checkContiguous(c, key_arg);

  map_.reserve(key.numel());

  const auto num_threads = at::get_num_threads();
  const auto grain_size = std::max(
      (key.numel() + num_threads - 1) / num_threads, at::internal::GRAIN_SIZE);
  const auto key_data = key.data_ptr<KeyType>();

  at::parallel_for(0, key.numel(), grain_size, [&](int64_t beg, int64_t end) {
    for (int64_t i = beg; i < end; ++i) {
      auto [iterator, inserted] = map_.insert({key_data[i], i});
      TORCH_CHECK(inserted, "Found duplicated key.");
    }
  });
};

template <typename KeyType>
at::Tensor CPUHashMap<KeyType>::get(const at::Tensor& query) {
  at::TensorArg query_arg{query, "query", 0};
  at::CheckedFrom c{"HashMap.get"};
  at::checkDeviceType(c, query, at::DeviceType::CPU);
  at::checkDim(c, query_arg, 1);
  at::checkContiguous(c, query_arg);

  const auto options = at::TensorOptions().dtype(at::kLong);
  const auto out = at::empty({query.numel()}, options);
  auto out_data = out.data_ptr<int64_t>();

  const auto num_threads = at::get_num_threads();
  const auto grain_size =
      std::max((query.numel() + num_threads - 1) / num_threads,
               at::internal::GRAIN_SIZE);
  const auto query_data = query.data_ptr<int64_t>();

  at::parallel_for(0, query.numel(), grain_size, [&](int64_t beg, int64_t end) {
    for (int64_t i = beg; i < end; ++i) {
      auto it = map_.find(query_data[i]);
      out_data[i] = (it != map_.end()) ? it->second : -1;
    }
  });

  return out;
}

TORCH_LIBRARY(pyg, m) {
  m.class_<CPUHashMap<int64_t>>("CPUHashMap")
      .def(torch::init<at::Tensor&>())
      .def("get", &CPUHashMap<int64_t>::get);
}

}  // namespace classes
}  // namespace pyg
