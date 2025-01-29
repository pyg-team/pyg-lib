#include <ATen/ATen.h>
#include <torch/library.h>
#include <cuco/static_map.cuh>
#include <limits>

namespace pyg {
namespace classes {

namespace {

#define DISPATCH_CASE_KEY(...)                         \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define DISPATCH_KEY(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_KEY(__VA_ARGS__))

struct HashMapImpl {
  virtual ~HashMapImpl() = default;
  virtual at::Tensor get(const at::Tensor& query) = 0;
  virtual at::Tensor keys() = 0;
};

template <typename KeyType>
struct CUDAHashMapImpl : HashMapImpl {
 public:
  using ValueType = int64_t;

  CUDAHashMapImpl(const at::Tensor& key) {
    KeyType constexpr empty_key_sentinel = std::numeric_limits<KeyType>::min();
    ValueType constexpr empty_value_sentinel = -1;

    map_ = std::make_unique<cuco::static_map<KeyType, ValueType>>(
        2 * key.numel(),  // load_factor = 0.5
        cuco::empty_key{empty_key_sentinel},
        cuco::empty_value{empty_value_sentinel});

    const auto key_data = key.data_ptr<KeyType>();
    const auto options =
        key.options().dtype(c10::CppTypeToScalarType<ValueType>::value);
    const auto value = at::arange(key.numel(), options);
    const auto value_data = value.data_ptr<ValueType>();
    const auto zipped =
        thrust::make_zip_iterator(thrust::make_tuple(key_data, value_data));

    map_->insert(zipped, zipped + key.numel());
  }

  at::Tensor get(const at::Tensor& query) override {
    const auto options =
        query.options().dtype(c10::CppTypeToScalarType<ValueType>::value);
    const auto out = at::empty({query.numel()}, options);
    const auto query_data = query.data_ptr<KeyType>();
    auto out_data = out.data_ptr<ValueType>();

    map_->find(query_data, query_data + query.numel(), out_data);

    return out;
  }

  at::Tensor keys() override {
    // TODO This will not work in multi-GPU scenarios.
    const auto options = at::TensorOptions().device(at::DeviceType::CUDA);
    const auto size = static_cast<int64_t>(map_->size());
    const auto key = at::empty(
        {size}, options.dtype(c10::CppTypeToScalarType<KeyType>::value));
    const auto value = at::empty(
        {size}, options.dtype(c10::CppTypeToScalarType<ValueType>::value));
    auto key_data = key.data_ptr<KeyType>();
    auto value_data = value.data_ptr<ValueType>();

    map_->retrieve_all(key_data, value_data);

    return key.index_select(0, value.argsort());
  }

 private:
  std::unique_ptr<cuco::static_map<KeyType, ValueType>> map_;
};

struct CUDAHashMap : torch::CustomClassHolder {
 public:
  CUDAHashMap(const at::Tensor& key) {
    at::TensorArg key_arg{key, "key", 0};
    at::CheckedFrom c{"CUDAHashMap.init"};
    at::checkDeviceType(c, key, at::DeviceType::CUDA);
    at::checkDim(c, key_arg, 1);
    at::checkContiguous(c, key_arg);

    DISPATCH_KEY(key.scalar_type(), "cuda_hash_map_init", [&] {
      map_ = std::make_unique<CUDAHashMapImpl<scalar_t>>(key);
    });
  }

  at::Tensor get(const at::Tensor& query) {
    at::TensorArg query_arg{query, "query", 0};
    at::CheckedFrom c{"CUDAHashMap.get"};
    at::checkDeviceType(c, query, at::DeviceType::CUDA);
    at::checkDim(c, query_arg, 1);
    at::checkContiguous(c, query_arg);

    return map_->get(query);
  }

  at::Tensor keys() { return map_->keys(); }

 private:
  std::unique_ptr<HashMapImpl> map_;
};

}  // namespace

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.class_<CUDAHashMap>("CUDAHashMap")
      .def(torch::init<at::Tensor&>())
      .def("get", &CUDAHashMap::get)
      .def("keys", &CUDAHashMap::keys)
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<CUDAHashMap>& self) -> at::Tensor {
            return self->keys();
          },
          // __setstate__
          [](const at::Tensor& state) -> c10::intrusive_ptr<CUDAHashMap> {
            return c10::make_intrusive<CUDAHashMap>(state);
          });
}

}  // namespace classes
}  // namespace pyg
