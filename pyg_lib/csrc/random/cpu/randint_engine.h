#include <torch/torch.h>

#include <limits.h>

namespace pyg {
namespace random {
const int RAND_PREFETCH_THRESHOLD = 128;
const int RAND_PREFETCH_BITS = 64;
template <typename T>
class RandintEngine {
 public:
  RandintEngine() : size_(RAND_PREFETCH_THRESHOLD), bits_(64) {
    prefetch_randint_ = torch::randint(
        0L, std::numeric_limits<int64_t>::max(), {RAND_PREFETCH_THRESHOLD},
        torch::TensorOptions().dtype(torch::kInt64));
  }
  template <unsigned B>
  T rand(T range) {
    T num;
    if (bits_ < B) {
      if (size_ > 0) {
        size_--;
        bits_ = 64;
      } else {
        prefetch_randint_ = torch::randint(
            0L, std::numeric_limits<int64_t>::max(), {RAND_PREFETCH_THRESHOLD},
            torch::TensorOptions().dtype(torch::kInt64));
        size_ = RAND_PREFETCH_THRESHOLD;
        bits_ = RAND_PREFETCH_BITS;
      }
    }
    int64_t* prefetch_ptr = prefetch_randint_.data_ptr<int64_t>();
    int64_t res = (prefetch_ptr[size_ - 1] % range) & ((1ULL << B) - 1);
    return (T)res;
  }

  T operator()(T beg, T end) {
    T range = end - beg;
    if (range <= (1 << 15)) {
      return rand<16>(range) + beg;
    } else if (range <= (1 << 31)) {
      return rand<32>(range) + beg;
    }
    return rand<63>(range) + beg;
  }

 private:
  torch::Tensor prefetch_randint_;
  int size_;
  int bits_;
};

}  // namespace random

}  // namespace pyg
