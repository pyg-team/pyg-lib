#pragma once

#include <torch/library.h>

namespace pyg {
namespace classes {

struct CPUHashMap : torch::CustomClassHolder {
 public:
  using KeyType = std::
      variant<bool, uint8_t, int8_t, int16_t, int32_t, int64_t, float, double>;

  CPUHashMap(const at::Tensor& key) {
    // TODO Assert 1-dim

    // clang-format off
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Bool,
    key.scalar_type(),
    "cpu_hash_map_init",
    [&] {
      const auto key_data = key.data_ptr<scalar_t>();
      for (int64_t i = 0; i < key.numel(); ++i) {
        // TODO Check that key does not yet exist.
        map_[key_data[i]] = i;
      }
    });
    // clang-format on
  };

  at::Tensor get(const at::Tensor& query) {
    // TODO Assert 1-dim

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
        // TODO Insert -1 if key does not exist.
        out_data[i] = map_[query_data[i]];
      }
    });
    // clang-format on

    return out;
  }

 private:
  std::unordered_map<KeyType, int64_t> map_;
};

TORCH_LIBRARY(pyg, m) {
  m.class_<CPUHashMap>("CPUHashMap")
      .def(torch::init<at::Tensor&>())
      .def("get", &CPUHashMap::get);
}

}  // namespace classes
}  // namespace pyg
