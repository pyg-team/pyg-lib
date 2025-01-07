#pragma once

#include <torch/library.h>

namespace pyg {
namespace classes {

template <typename T>
struct CPUHashMap : torch::CustomClassHolder {
  std::unordered_map<T, int64_t> map;

  CPUHashMap(const at::Tensor& key) {
    // TODO Assert 1-dim
    const auto key_data = key.data_ptr<T>();
    for (int64_t i = 0; i < key.numel(); ++i) {
      // TODO Check that key does not yet exist.
      map[key_data[i]] = i;
    }
  };

  at::Tensor get(const at::Tensor& query) {
    // TODO Assert 1-dim
    const auto options = at::TensorOptions().dtype(at::kLong);
    auto out = at::empty({query.numel()}, options);

    const auto query_data = query.data_ptr<T>();
    auto out_data = out.data_ptr<int64_t>();

    for (size_t i = 0; i < query.numel(); ++i) {
      // TODO Insert -1 if key does not exist.
      out_data[i] = map[query_data[i]];
    }
    return out;
  }
};

TORCH_LIBRARY(pyg, m) {
  m.class_<CPUHashMap<int64_t>>("CPULongHashMap")
      .def(torch::init<at::Tensor&>())
      .def("get", &CPUHashMap<int64_t>::get);
}

}  // namespace classes
}  // namespace pyg
