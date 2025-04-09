#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>
#include <limits>

#ifndef _WIN32
#include <cuco/static_map.cuh>
#endif

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
  virtual int64_t size() = 0;
  virtual at::ScalarType dtype() = 0;
  virtual at::Device device() = 0;
};

#ifndef _WIN32
template <typename KeyType>
struct CUDAHashMapImpl : HashMapImpl {
 public:
  using ValueType = int64_t;

  CUDAHashMapImpl(const at::Tensor& key, double load_factor)
      : device_(key.device()) {
    cudaSetDevice(key.get_device());

    KeyType constexpr empty_key_sentinel = std::numeric_limits<KeyType>::min();
    ValueType constexpr empty_value_sentinel = -1;

    size_t capacity = std::ceil(key.numel() / load_factor);
    map_ = std::make_unique<cuco::static_map<KeyType, ValueType>>(
        capacity, cuco::empty_key{empty_key_sentinel},
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
    cudaSetDevice(query.get_device());

    const auto options =
        query.options().dtype(c10::CppTypeToScalarType<ValueType>::value);
    const auto out = at::empty({query.numel()}, options);
    const auto query_data = query.data_ptr<KeyType>();
    const auto out_data = out.data_ptr<ValueType>();

    map_->find(query_data, query_data + query.numel(), out_data);

    return out;
  }

  at::Tensor keys() override {
    cudaSetDevice(device_.index());

    const auto options = at::TensorOptions().device(device_);
    const at::Tensor key = at::empty({size()}, options.dtype(dtype()));
    const at::Tensor value = at::empty(
        {size()}, options.dtype(c10::CppTypeToScalarType<ValueType>::value));
    const auto key_data = key.data_ptr<KeyType>();
    const auto value_data = value.data_ptr<ValueType>();

    map_->retrieve_all(key_data, value_data);

    const auto perm = at::empty_like(value);
    perm.scatter_(0, value, at::arange(value.numel(), value.options()));

    return key.index_select(0, perm);
  }

  int64_t size() override { return static_cast<int64_t>(map_->size()); }

  at::ScalarType dtype() override {
    if (std::is_same<KeyType, int16_t>::value) {
      return at::kShort;
    } else if (std::is_same<KeyType, int32_t>::value) {
      return at::kInt;
    } else {
      return at::kLong;
    }
  }

  at::Device device() override { return device_; }

 private:
  std::unique_ptr<cuco::static_map<KeyType, ValueType>> map_;
  at::Device device_;
};
#endif

struct CUDAHashMap : torch::CustomClassHolder {
 public:
  CUDAHashMap(const at::Tensor& key, double load_factor = 0.5) {
#ifndef _WIN32
    at::TensorArg key_arg{key, "key", 0};
    at::CheckedFrom c{"CUDAHashMap.init"};
    at::checkDeviceType(c, key, at::DeviceType::CUDA);
    at::checkDim(c, key_arg, 1);
    at::checkContiguous(c, key_arg);

    DISPATCH_KEY(key.scalar_type(), "cuda_hash_map_init", [&] {
      map_ = std::make_unique<CUDAHashMapImpl<scalar_t>>(key, load_factor);
    });
#else
    TORCH_CHECK(false, "'CUDAHashMap' not supported on Windows");
#endif
  }

  at::Tensor get(const at::Tensor& query) {
#ifndef _WIN32
    at::TensorArg query_arg{query, "query", 0};
    at::CheckedFrom c{"CUDAHashMap.get"};
    at::checkDeviceType(c, query, at::DeviceType::CUDA);
    at::checkDim(c, query_arg, 1);
    at::checkContiguous(c, query_arg);

    return map_->get(query);
#else
    TORCH_CHECK(false, "'CUDAHashMap' not supported on Windows");
#endif
  }

  at::Tensor keys() {
#ifndef _WIN32
    return map_->keys();
#else
    TORCH_CHECK(false, "'CUDAHashMap' not supported on Windows");
#endif
  }

  int64_t size() {
#ifndef _WIN32
    return map_->size();
#else
    TORCH_CHECK(false, "'CUDAHashMap' not supported on Windows");
#endif
  }

  at::ScalarType dtype() {
#ifndef _WIN32
    return map_->dtype();
#else
    TORCH_CHECK(false, "'CUDAHashMap' not supported on Windows");
#endif
  }

  at::Device device() {
#ifndef _WIN32
    return map_->device();
#else
    TORCH_CHECK(false, "'CUDAHashMap' not supported on Windows");
#endif
  }

 private:
#ifndef _WIN32
  std::unique_ptr<HashMapImpl> map_;
#endif
};

}  // namespace

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.class_<CUDAHashMap>("CUDAHashMap")
      .def(torch::init<at::Tensor&, double>())
      .def("get", &CUDAHashMap::get)
      .def("keys", &CUDAHashMap::keys)
      .def("size", &CUDAHashMap::size)
      .def("dtype", &CUDAHashMap::dtype)
      .def("device", &CUDAHashMap::device)
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
