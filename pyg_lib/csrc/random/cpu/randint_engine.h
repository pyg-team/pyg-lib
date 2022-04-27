#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

#include <limits.h>

namespace pyg {
namespace random {

const int RAND_PREFETCH_SIZE = 128;

// Use at::randint to generate 64-bit random numbers
const int RAND_PREFETCH_BITS = 64;

/**
 * Amortized O(1) uniform random within a range.
 *
 * Reduced torch random overheads by prefetching large random bits in advance.
 * Prefetched random bits are consumed on demand.
 *
 */
class PrefetchedRandint {
 public:
  PrefetchedRandint()
      : PrefetchedRandint(RAND_PREFETCH_SIZE, RAND_PREFETCH_BITS) {}
  PrefetchedRandint(int size, int bits) { prefetch(size, bits); }

  /**
   * Generate next random number in a range uniformly
   *
   * @tparam T user type of random numbers
   *
   * @param range the range of uniform distribution
   *
   * @returns the chosen number in that range
   */
  template <typename T>
  T next(T range) {
    unsigned needed = 64;

    // Mutiple levels of range to save prefetched bits.
    if (range < (1 << 16)) {
      needed = 16;
    } else if (range < (1UL << 32)) {
      needed = 32;
    }

    // Consume some bits
    if (bits_ < needed) {
      if (size_ > 0) {
        size_--;
        bits_ = RAND_PREFETCH_BITS;
      } else {
        // Prefetch if no enough bits
        prefetch(prefetched_randint_.size(0), RAND_PREFETCH_BITS);
      }
    }

    // Currently torch could only make 64-bit signed random numbers.
    uint64_t* prefetch_ptr =
        reinterpret_cast<uint64_t*>(prefetched_randint_.data_ptr<int64_t>());

    // Take the lower bits of current 64-bit number to fit in the range.
    uint64_t mask = (needed == 64) ? std::numeric_limits<uint64_t>::max()
                                   : ((1ULL << needed) - 1);
    uint64_t res = (prefetch_ptr[size_] & mask) % range;

    // Update the state
    prefetch_ptr[size_] >>= needed;
    bits_ -= needed;
    return (T)res;
  }

 private:
  // Prefetch random bits. In-place random if prefetching size is the same.
  void prefetch(int size, int bits) {
    if (prefetched_randint_.size(0) != size) {
      prefetched_randint_ =
          at::randint(std::numeric_limits<int64_t>::min(),
                      std::numeric_limits<int64_t>::max(), {size},
                      at::TensorOptions().dtype(at::kLong));
    } else {
      prefetched_randint_.random_(std::numeric_limits<int64_t>::min(),
                                  std::numeric_limits<int64_t>::max());
    }
    size_ = size - 1;
    bits_ = bits;
  }

  at::Tensor prefetched_randint_;
  int size_;
  int bits_;
};

/**
 * Randint functor for uniform integer distribution.
 * Wrapped PrefetchedRandint as its efficient core implementation.
 */
template <typename T>
class RandintEngine {
 public:
  RandintEngine() : prefetched_(RAND_PREFETCH_SIZE, RAND_PREFETCH_BITS) {}

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
