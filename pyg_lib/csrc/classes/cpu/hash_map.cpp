#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
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
  virtual int64_t size() = 0;
  virtual at::ScalarType dtype() = 0;
};

template <typename KeyType>
struct CPUHashMapImpl : HashMapImpl {
 public:
  using ValueType = int64_t;

  CPUHashMapImpl(const at::Tensor& key) {
    map_.reserve(key.numel());

    const auto key_data = key.data_ptr<KeyType>();
    for (int64_t i = 0; i < key.numel(); ++i) {
      const auto [iterator, inserted] = map_.insert({key_data[i], i});
      TORCH_CHECK(inserted, "Found duplicated key in 'HashMap'.");
    }
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
    const at::Tensor key =
        at::empty({size()}, at::TensorOptions().dtype(dtype()));
    const auto key_data = key.data_ptr<KeyType>();

    for (const auto& pair : map_) {  // No efficient multi-threading possible :(
      key_data[pair.second] = pair.first;
    }

    return key;
  }

  int64_t size() override { return static_cast<int64_t>(map_.size()); }

  at::ScalarType dtype() override {
    if (std::is_same<KeyType, int16_t>::value) {
      return at::kShort;
    } else if (std::is_same<KeyType, int32_t>::value) {
      return at::kInt;
    } else {
      return at::kLong;
    }
  }

 private:
  phmap::flat_hash_map<KeyType, ValueType> map_;
};

template <typename KeyType, size_t num_submaps>
struct ParallelCPUHashMapImpl : HashMapImpl {
 public:
  using ValueType = int64_t;

  ParallelCPUHashMapImpl(const at::Tensor& key) {
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
    const at::Tensor key =
        at::empty({size()}, at::TensorOptions().dtype(dtype()));
    const auto key_data = key.data_ptr<KeyType>();

    for (const auto& pair : map_) {  // No efficient multi-threading possible :(
      key_data[pair.second] = pair.first;
    }

    return key;
  }

  int64_t size() override { return static_cast<int64_t>(map_.size()); }

  at::ScalarType dtype() override {
    if (std::is_same<KeyType, int16_t>::value) {
      return at::kShort;
    } else if (std::is_same<KeyType, int32_t>::value) {
      return at::kInt;
    } else {
      return at::kLong;
    }
  }

 private:
  phmap::parallel_flat_hash_map<
      KeyType,
      ValueType,
      phmap::priv::hash_default_hash<KeyType>,
      phmap::priv::hash_default_eq<KeyType>,
      phmap::priv::Allocator<std::pair<const KeyType, ValueType>>,
      num_submaps>
      map_;
};

struct CPUHashMap : torch::CustomClassHolder {
 public:
  CPUHashMap(const at::Tensor& key, int64_t num_submaps = 0) {
    at::TensorArg key_arg{key, "key", 0};
    at::CheckedFrom c{"CPUHashMap.init"};
    at::checkDeviceType(c, key, at::DeviceType::CPU);
    at::checkDim(c, key_arg, 1);
    at::checkContiguous(c, key_arg);

    DISPATCH_KEY(key.scalar_type(), "cpu_hash_map_init", [&] {
      switch (num_submaps) {
        case -1:  // Auto-infer:
          if (key.numel() < 200'000) {
            map_ = std::make_unique<CPUHashMapImpl<scalar_t>>(key);
          } else {
            map_ = std::make_unique<ParallelCPUHashMapImpl<scalar_t, 8>>(key);
          }
          break;
        case 0:
          map_ = std::make_unique<CPUHashMapImpl<scalar_t>>(key);
          break;
        case 2:
          map_ = std::make_unique<ParallelCPUHashMapImpl<scalar_t, 1>>(key);
          break;
        case 4:
          map_ = std::make_unique<ParallelCPUHashMapImpl<scalar_t, 2>>(key);
          break;
        case 8:
          map_ = std::make_unique<ParallelCPUHashMapImpl<scalar_t, 3>>(key);
          break;
        case 16:
          map_ = std::make_unique<ParallelCPUHashMapImpl<scalar_t, 4>>(key);
          break;
        case 32:
          map_ = std::make_unique<ParallelCPUHashMapImpl<scalar_t, 5>>(key);
          break;
        case 64:
          map_ = std::make_unique<ParallelCPUHashMapImpl<scalar_t, 6>>(key);
          break;
        case 128:
          map_ = std::make_unique<ParallelCPUHashMapImpl<scalar_t, 7>>(key);
          break;
        case 256:
          map_ = std::make_unique<ParallelCPUHashMapImpl<scalar_t, 8>>(key);
          break;
        case 512:
          map_ = std::make_unique<ParallelCPUHashMapImpl<scalar_t, 9>>(key);
          break;
        case 1024:
          map_ = std::make_unique<ParallelCPUHashMapImpl<scalar_t, 10>>(key);
          break;
        case 2048:
          map_ = std::make_unique<ParallelCPUHashMapImpl<scalar_t, 11>>(key);
          break;
        case 4096:
          map_ = std::make_unique<ParallelCPUHashMapImpl<scalar_t, 12>>(key);
          break;
        default:
          TORCH_CHECK(false, "'num_submaps' needs to be a power of 2");
      }
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

  int64_t size() { return map_->size(); }

  at::ScalarType dtype() { return map_->dtype(); }

  at::Device device() { return at::Device(at::kCPU); }

 private:
  std::unique_ptr<HashMapImpl> map_;
};

}  // namespace

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.class_<CPUHashMap>("CPUHashMap")
      .def(torch::init<at::Tensor&, int64_t>())
      .def("get", &CPUHashMap::get)
      .def("keys", &CPUHashMap::keys)
      .def("size", &CPUHashMap::size)
      .def("dtype", &CPUHashMap::dtype)
      .def("device", &CPUHashMap::device)
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
