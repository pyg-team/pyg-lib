#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <parallel_hashmap/phmap.h>
#include <torch/library.h>

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
struct CPUHashMapImpl : HashMapImpl {
 public:
  using ValueType = int64_t;

  CPUHashMapImpl(const at::Tensor& key) {
    map_.reserve(key.numel());

    const auto key_data = key.data_ptr<KeyType>();

    const auto num_threads = at::get_num_threads();
    const auto grain_size =
        std::max((key.numel() + num_threads - 1) / num_threads,
                 at::internal::GRAIN_SIZE);

    at::parallel_for(0, key.numel(), grain_size, [&](int64_t beg, int64_t end) {
      for (int64_t i = beg; i < end; ++i) {
        const auto [iterator, inserted] = map_.insert({key_data[i], i});
        TORCH_CHECK(inserted, "Found duplicated key in 'HashMap'.");
      }
    });
  }

  at::Tensor get(const at::Tensor& query) override {
    const auto options =
        query.options().dtype(c10::CppTypeToScalarType<ValueType>::value);
    const auto out = at::empty({query.numel()}, options);
    const auto query_data = query.data_ptr<KeyType>();
    const auto out_data = out.data_ptr<ValueType>();

    const auto num_threads = at::get_num_threads();
    const auto grain_size =
        std::max((query.numel() + num_threads - 1) / num_threads,
                 at::internal::GRAIN_SIZE);

    at::parallel_for(0, query.numel(), grain_size, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i) {
        const auto it = map_.find(query_data[i]);
        out_data[i] = (it != map_.end()) ? it->second : -1;
      }
    });

    return out;
  }

  at::Tensor keys() override {
    const auto size = static_cast<int64_t>(map_.size());

    at::Tensor key;
    if (std::is_same<KeyType, int16_t>::value) {
      key = at::empty({size}, at::TensorOptions().dtype(at::kShort));
    } else if (std::is_same<KeyType, int32_t>::value) {
      key = at::empty({size}, at::TensorOptions().dtype(at::kInt));
    } else {
      key = at::empty({size}, at::TensorOptions().dtype(at::kLong));
    }
    const auto key_data = key.data_ptr<KeyType>();

    for (const auto& pair : map_) {  // No efficient multi-threading possible :(
      key_data[pair.second] = pair.first;
    }

    return key;
  }

 private:
  phmap::parallel_flat_hash_map<
      KeyType,
      ValueType,
      phmap::priv::hash_default_hash<KeyType>,
      phmap::priv::hash_default_eq<KeyType>,
      phmap::priv::Allocator<std::pair<const KeyType, ValueType>>,
      12,
      std::mutex>
      map_;
};

struct CPUHashMap : torch::CustomClassHolder {
 public:
  CPUHashMap(const at::Tensor& key) {
    at::TensorArg key_arg{key, "key", 0};
    at::CheckedFrom c{"CPUHashMap.init"};
    at::checkDeviceType(c, key, at::DeviceType::CPU);
    at::checkDim(c, key_arg, 1);
    at::checkContiguous(c, key_arg);

    DISPATCH_KEY(key.scalar_type(), "cpu_hash_map_init", [&] {
      map_ = std::make_unique<CPUHashMapImpl<scalar_t>>(key);
    });
  }

  at::Tensor get(const at::Tensor& query) {
    at::TensorArg query_arg{query, "query", 0};
    at::CheckedFrom c{"CPUHashMap.get"};
    at::checkDeviceType(c, query, at::DeviceType::CPU);
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
  m.class_<CPUHashMap>("CPUHashMap")
      .def(torch::init<at::Tensor&>())
      .def("get", &CPUHashMap::get)
      .def("keys", &CPUHashMap::keys)
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<CPUHashMap>& self) -> at::Tensor {
            return self->keys();
          },
          // __setstate__
          [](const at::Tensor& state) -> c10::intrusive_ptr<CPUHashMap> {
            return c10::make_intrusive<CPUHashMap>(state);
          });
}

}  // namespace classes
}  // namespace pyg
