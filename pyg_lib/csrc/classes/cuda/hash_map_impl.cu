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

  CUDAHashMapImpl(const at::Tensor& key) {}

  at::Tensor get(const at::Tensor& query) override { return query; }
};

template struct CUDAHashMapImpl<bool>;
template struct CUDAHashMapImpl<uint8_t>;
template struct CUDAHashMapImpl<int8_t>;
template struct CUDAHashMapImpl<int16_t>;
template struct CUDAHashMapImpl<int32_t>;
template struct CUDAHashMapImpl<int64_t>;
template struct CUDAHashMapImpl<float>;
template struct CUDAHashMapImpl<double>;

}  // namespace

}  // namespace classes
}  // namespace pyg
