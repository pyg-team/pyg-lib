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
    KeyType constexpr empty_key_sentinel = -1;  // TODO
    ValueType constexpr empty_value_sentinel = -1;

    map_ = std::make_unique<cuco::static_map<KeyType, ValueType>>(
        2 * key.numel(),  // loader_factor = 0.5
        cuco::empty_key{empty_key_sentinel},
        cuco::empty_value{empty_value_sentinel});

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
    const auto query_data = query.data_ptr<KeyType>();
    auto out_data = out.data_ptr<int64_t>();

    map_->find(query_data, out_data, query.numel());

    return out;
  }

 private:
  std::unique_ptr<cuco::static_map<KeyType, ValueType>> map_;
};

// template struct CUDAHashMapImpl<bool>;
// template struct CUDAHashMapImpl<uint8_t>;
// template struct CUDAHashMapImpl<int8_t>;
// template struct CUDAHashMapImpl<int16_t>;
// template struct CUDAHashMapImpl<int32_t>;
// template struct CUDAHashMapImpl<int64_t>;
// template struct CUDAHashMapImpl<float>;
// template struct CUDAHashMapImpl<double>;

struct CUDAHashMap : torch::CustomClassHolder {
 public:
  CUDAHashMap(const at::Tensor& key) {}

  at::Tensor get(const at::Tensor& query) { return query; }
};

}  // namespace

}  // namespace classes
}  // namespace pyg
