#pragma once

#include <torch/torch.h>

#include <limits.h>

namespace pyg {
namespace random {

const int RAND_PREFETCH_THRESHOLD = 128;

// Use torch::randint to generate 64-bit random numbers
const int RAND_PREFETCH_BITS = 64;

class PrefetchedRandint {
 public:
  PrefetchedRandint(int size = RAND_PREFETCH_THRESHOLD,
                    int bits = RAND_PREFETCH_BITS) {
    prefetch_();
  }

  template <typename T>
  T next(T range) {
    unsigned needed = 64;

    // Mutiple levels of range to save prefetched bits
    if (range <= (1 << 15)) {
      needed = 16;
    } else if (range <= (1UL << 31)) {
      needed = 32;
    }

    if (bits_ < needed) {
      if (size_ > 0) {
        size_--;
        bits_ = RAND_PREFETCH_BITS;
      } else {
        // Prefetch if no enough bits
        prefetch_();
      }
    }

    // Currently torch could only make 64-bit signed numbers
    uint64_t* prefetch_ptr =
        reinterpret_cast<uint64_t*>(prefetched_randint_.data_ptr<int64_t>());
    uint64_t mask = (needed == 64) ? std::numeric_limits<uint64_t>::max()
                                   : (1ULL << needed) - 1;
    uint64_t res = (prefetch_ptr[size_ - 1] & mask) % range;
    prefetch_ptr[size_ - 1] >>= needed;
    bits_ -= needed;
    return (T)res;
  }

 private:
  void prefetch_() {
    prefetched_randint_ = torch::randint(
        std::numeric_limits<int64_t>::min(),
        std::numeric_limits<int64_t>::max(), {RAND_PREFETCH_THRESHOLD},
        torch::TensorOptions().dtype(torch::kInt64));
    size_ = RAND_PREFETCH_THRESHOLD;
    bits_ = RAND_PREFETCH_BITS;
  }

  torch::Tensor prefetched_randint_;
  int size_;
  int bits_;
};

template <typename T>
class RandintEngine {
 public:
  RandintEngine() : prefetched_(RAND_PREFETCH_THRESHOLD, RAND_PREFETCH_BITS) {}

  // Uniform random number within range [beg, end)
  T operator()(T beg, T end) {
    TORCH_CHECK(beg < end, "Randint engine illegal range");

    T range = end - beg;
    return prefetched_.next(range) + beg;
  }

 private:
  PrefetchedRandint prefetched_;
};

}  // namespace random

}  // namespace pyg
