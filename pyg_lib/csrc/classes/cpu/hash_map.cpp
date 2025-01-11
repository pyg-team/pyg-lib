#include "hash_map.h"

#include <torch/library.h>

namespace pyg {
namespace classes {

CPUHashMap::CPUHashMap(const at::Tensor& key) {
  at::TensorArg key_arg{key, "key", 0};
  at::CheckedFrom c{"HashMap.init"};
  at::checkDeviceType(c, key, at::DeviceType::CPU);
  at::checkDim(c, key_arg, 1);
  at::checkContiguous(c, key_arg);

  // clang-format off
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Bool,
    key.scalar_type(),
    "cpu_hash_map_init",
    [&] {
      const auto key_data = key.data_ptr<scalar_t>();
      for (int64_t i = 0; i < key.numel(); ++i) {
        auto [iterator, inserted] = map_.insert({key_data[i], i});
        TORCH_CHECK(inserted, "Found duplicated key.");
      }
    });
  // clang-format on
};

at::Tensor CPUHashMap::get(const at::Tensor& query) {
  at::TensorArg query_arg{query, "query", 0};
  at::CheckedFrom c{"HashMap.get"};
  at::checkDeviceType(c, query, at::DeviceType::CPU);
  at::checkDim(c, query_arg, 1);
  at::checkContiguous(c, query_arg);

  const auto options = at::TensorOptions().dtype(at::kLong);
  const auto out = at::empty({query.numel()}, options);
  auto out_data = out.data_ptr<int64_t>();

  // clang-format off
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Bool,
    query.scalar_type(),
    "cpu_hash_map_get",
    [&] {
      const auto query_data = query.data_ptr<scalar_t>();

      for (size_t i = 0; i < query.numel(); ++i) {
        auto it = map_.find(query_data[i]);
        out_data[i] = (it != map_.end()) ? it->second : -1;
      }
    });
  // clang-format on

  return out;
}

TORCH_LIBRARY(pyg, m) {
  m.class_<CPUHashMap>("CPUHashMap")
      .def(torch::init<at::Tensor&>())
      .def("get", &CPUHashMap::get);
}

}  // namespace classes
}  // namespace pyg
