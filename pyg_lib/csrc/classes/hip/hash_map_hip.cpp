// ROCm/HIP implementation of GPU HashMap
// Uses sorted array + binary search via ATen (not thrust directly)
// This avoids the cuda/cccl header conflicts with rocThrust

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
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
  virtual at::Device device() = 0;
};

// HIP implementation using sorted arrays + binary search (ATen-only, no thrust)
template <typename KeyType>
struct HIPHashMapImpl : HashMapImpl {
 public:
  using ValueType = int64_t;

  HIPHashMapImpl(const at::Tensor& key, double load_factor)
      : device_(key.device()) {
    // Store sorted keys and their original indices
    const auto options = key.options();
    const auto value_options =
        options.dtype(c10::CppTypeToScalarType<ValueType>::value);

    // Create value tensor (indices 0 to N-1)
    sorted_values_ = at::arange(key.numel(), value_options);

    // Clone keys and sort together with values
    sorted_keys_ = key.clone();

    // Sort by keys, permuting values accordingly
    auto sort_result = at::sort(sorted_keys_);
    sorted_keys_ = std::get<0>(sort_result);
    auto sort_indices = std::get<1>(sort_result);
    sorted_values_ = sorted_values_.index_select(0, sort_indices);
  }

  at::Tensor get(const at::Tensor& query) override {
    // Use searchsorted to find positions, then verify matches
    auto positions = at::searchsorted(sorted_keys_, query);
    auto out = sorted_values_.new_full({query.numel()}, -1);

    // Clamp positions to valid range
    positions = at::clamp(positions, 0, sorted_keys_.numel() - 1);

    // Get keys at found positions
    auto found_keys = sorted_keys_.index_select(0, positions);

    // Create mask where query matches found key
    auto mask = (found_keys == query);

    // Get values where mask is true
    auto found_values = sorted_values_.index_select(0, positions);
    out = at::where(mask, found_values, out);

    return out;
  }

  at::Tensor keys() override {
    // Return keys in original order (unsort using values as indices)
    auto perm = at::empty_like(sorted_values_);
    perm.scatter_(0, sorted_values_,
                  at::arange(sorted_values_.numel(), sorted_values_.options()));
    return sorted_keys_.index_select(0, perm);
  }

  int64_t size() override { return sorted_keys_.numel(); }

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
  at::Tensor sorted_keys_;
  at::Tensor sorted_values_;
  at::Device device_;
};

struct HIPHashMap : torch::CustomClassHolder {
 public:
  HIPHashMap(const at::Tensor& key, double load_factor = 0.5) {
    at::TensorArg key_arg{key, "key", 0};
    at::CheckedFrom c{"HIPHashMap.init"};
    at::checkDeviceType(c, key, at::DeviceType::CUDA);  // CUDA type for ROCm
    at::checkDim(c, key_arg, 1);
    at::checkContiguous(c, key_arg);

    DISPATCH_KEY(key.scalar_type(), "hip_hash_map_init", [&] {
      map_ = std::make_unique<HIPHashMapImpl<scalar_t>>(key, load_factor);
    });
  }

  at::Tensor get(const at::Tensor& query) {
    at::TensorArg query_arg{query, "query", 0};
    at::CheckedFrom c{"HIPHashMap.get"};
    at::checkDeviceType(c, query, at::DeviceType::CUDA);
    at::checkDim(c, query_arg, 1);
    at::checkContiguous(c, query_arg);

    return map_->get(query);
  }

  at::Tensor keys() { return map_->keys(); }
  int64_t size() { return map_->size(); }
  at::ScalarType dtype() { return map_->dtype(); }
  at::Device device() { return map_->device(); }

 private:
  std::unique_ptr<HashMapImpl> map_;
};

}  // namespace

// Note: For ROCm, we register CUDAHashMap but use HIP implementation
// The dispatch key "CUDA" works for both CUDA and ROCm in PyTorch
TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.class_<HIPHashMap>("CUDAHashMap")
      .def(torch::init<at::Tensor&, double>())
      .def("get", &HIPHashMap::get)
      .def("keys", &HIPHashMap::keys)
      .def("size", &HIPHashMap::size)
      .def("dtype", &HIPHashMap::dtype)
      .def("device", &HIPHashMap::device)
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<HIPHashMap>& self) -> at::Tensor {
            return self->keys();
          },
          // __setstate__
          [](const at::Tensor& state) -> c10::intrusive_ptr<HIPHashMap> {
            return c10::make_intrusive<HIPHashMap>(state);
          });
}

}  // namespace classes
}  // namespace pyg
