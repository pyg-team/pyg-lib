#include "hash_map.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

#include "cpu/hash_map_impl.cpp"

namespace pyg {
namespace classes {

HashMap::HashMap(const at::Tensor& key) {
  at::TensorArg key_arg{key, "key", 0};
  at::CheckedFrom c{"HashMap.init"};
  at::checkDeviceType(c, key, at::DeviceType::CPU);
  at::checkDim(c, key_arg, 1);
  at::checkContiguous(c, key_arg);

  map_ = get_hash_map(key);

  /* static auto op = c10::Dispatcher::singleton() */
  /*                      .findSchemaOrThrow("pyg::get_hash_map", "") */
  /*                      .typed<HashMapImpl*(at::Tensor)>(); */
  /* map_ = op.call(key); */

  // clang-format off
  /* AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Bool, */
  /* key.scalar_type(), */
  /* "hash_map_init", */
  /* [&] { */
  /*   /1* if (key.is_cpu) { *1/ */
  /*   map_ = std::make_unique<CPUHashMapImpl<scalar_t>>(key); */
  /*   /1* } else { *1/ */
  /*   /1*   AT_ERROR("Received invalid device type for 'HashMap'."); *1/ */
  /*   /1* } *1/ */
  /* }); */
  // clang-format on
}

at::Tensor HashMap::get(const at::Tensor& query) {
  at::TensorArg query_arg{query, "query", 0};
  at::CheckedFrom c{"HashMap.get"};
  at::checkDeviceType(c, query, at::DeviceType::CPU);
  at::checkDim(c, query_arg, 1);
  at::checkContiguous(c, query_arg);

  return map_->get(query);
}

TORCH_LIBRARY(pyg, m) {
  m.class_<HashMap>("HashMap")
      .def(torch::init<at::Tensor&>())
      .def("get", &HashMap::get);
}

}  // namespace classes
}  // namespace pyg
