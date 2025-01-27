#include <ATen/ATen.h>
#include <cuco/static_map.cuh>

#include "../hash_map_impl.h"

namespace pyg {
namespace classes {

namespace {

template <typename KeyType>
struct CUDAHashMapImpl : HashMapImpl {
 public:
  using ValueType = int64_t;

  CUDAHashMapImpl(const at::Tensor& key) {
    map_ = std::make_unique<cuco::static_map<KeyType, ValueType>>(
        2 * key.numel(),      // loader_factor = 0.5
        cuco::empty_key{-1},  // TODO
        cuco::empty_value{-1});

    const auto options =
        at::TensorOptions().device(key.device()).dtype(at::kLong);
    const auto value = at::arange(key.numel(), options);
    const auto key_data = key.data_ptr<KeyType>();
    const auto value_data = value.data_ptr<ValueType>();

    map_->insert(key_data, value_data, key.numel());
  }

  at::Tensor get(const at::Tensor& query) override {
    const auto options =
        at::TensorOptions().device(query.device()).dtype(at::kLong);
    const auto out = at::empty({query.numel()}, options);
    const auto query_data = query_data.data_ptr<KeyType>();
    auto out_data = out.data_ptr<int64_t>();

    map_->find(query_data, out_data, query.numel());

    return out;
  }

 private:
  std::unique_ptr<cuco::static_map<KeyType, ValueType>> map_;
};

// template struct CUDAHashMapImpl<bool>;
// template struct CUDAHashMapImpl<uint8_t>;
template struct CUDAHashMapImpl<int8_t>;
template struct CUDAHashMapImpl<int16_t>;
template struct CUDAHashMapImpl<int32_t>;
template struct CUDAHashMapImpl<int64_t>;
template struct CUDAHashMapImpl<float>;
template struct CUDAHashMapImpl<double>;

struct CUDAHashMap : torch::CustomClassHolder {
 public:
  CUDAHashMap(const at::Tensor& key) {}

  at::Tensor get(const at::Tensor& query) { return query; }
};

c10::intrusive_ptr<HashMapImpl> get_hash_map(const at::Tensor& key) {
  return c10::make_intrusive<CUDAHashMapImpl<int64_t>>(key);
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::get_hash_map"),
         TORCH_FN(sampled_op_kernel));
}

}  // namespace classes
}  // namespace pyg
