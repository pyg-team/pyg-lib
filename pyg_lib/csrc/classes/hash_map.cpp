#include "hash_map.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

#include "cpu/hash_map_impl.cpp"
#ifdef __CUDACC__
#include "cuda/hash_map_impl.cu"
#endif

namespace pyg {
namespace classes {

HashMap::HashMap(const at::Tensor& key) {
  at::TensorArg key_arg{key, "key", 0};
  at::CheckedFrom c{"HashMap.init"};
  /* at::checkDeviceType(c, key, at::DeviceType::CPU); */
  at::checkDim(c, key_arg, 1);
  at::checkContiguous(c, key_arg);

  if (key.is_cpu()) {
    std::cout << "CPU" << std::endl;
    map_ = std::make_unique<CPUHashMapImpl<int64_t>>(key);
#ifdef __CUDACC__
  } else if (key.is_cuda()) {
    std::cout << "CUDA" << std::endl;
    map_ = std::make_unique<CUDAHashMapImpl<int64_t>>(key);
#endif
  } else {
    AT_ERROR("Received unsupported device type for 'HashMap'.");
  }

  /* static auto op = */
  /*     c10::Dispatcher::singleton() */
  /*         .findSchemaOrThrow("pyg::get_hash_map", "") */
  /*         .typed<c10::intrusive_ptr<HashMapImpl>(at::Tensor const&)>(); */
  /* op.call(key); */

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

/* HashMap::~HashMap() { */
/*   delete map_; */
/* } */

at::Tensor HashMap::get(const at::Tensor& query) {
  at::TensorArg query_arg{query, "query", 0};
  at::CheckedFrom c{"HashMap.get"};
  /* at::checkDeviceType(c, query, at::DeviceType::CPU); */
  at::checkDim(c, query_arg, 1);
  at::checkContiguous(c, query_arg);

  return query;

  /* return map_->get(query); */
}

/* TORCH_LIBRARY_FRAGMENT(pyg, m) { */
/*   m.def(TORCH_SELECTIVE_SCHEMA("pyg::get_hash_map(Tensor key) -> int")); */
/* } */

TORCH_LIBRARY(pyg, m) {
  m.class_<HashMap>("HashMap")
      .def(torch::init<at::Tensor&>())
      .def("get", &HashMap::get);
}

}  // namespace classes
}  // namespace pyg
