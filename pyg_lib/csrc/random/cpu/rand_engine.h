#pragma once

#include <ATen/ATen.h>
#include <limits.h>

#include "pyg_lib/csrc/config.h"
#if WITH_MKL_BLAS()
#include <mkl.h>
#endif

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
  RandintEngine() {
#if WITH_MKL_BLAS()
    vslNewStream(&stream_, VSL_BRNG_MT19937, 1);
#endif
  }
  ~RandintEngine() {
#if WITH_MKL_BLAS()
    vslDeleteStream(&stream_);
#endif
  }

  // Uniform random number within range [beg, end)
  T operator()(T beg, T end) {
    TORCH_CHECK(beg < end, "Randint engine illegal range");

    const T range = end - beg;
    return prefetched_.next(range) + beg;
  }

  // Generates `count` numbers within range [beg, end). If
  // possible, it will use specialized MKL implementation.
  // It is user's responsibility to ensure that `beg` and `end`
  // fit into int.
  std::vector<int> generate_range_of_ints(T beg, T end, int64_t count) {
    TORCH_CHECK(beg < end, "Randint engine illegal range");

    std::vector<int> result(count);
    const auto fallback_func = [this](T beg, T end, std::vector<int>& dst) {
      for (auto& val : dst)
        val = static_cast<int>((*this)(beg, end));
    };
    // #if WITH_MKL_BLAS()
    // const bool use_fallback_func = count >
    // std::numeric_limits<MKL_INT>::max(); if (use_fallback_func) {
    //   fallback_func(beg, end, result);
    // } else {
    //   const auto b = static_cast<int>(beg);
    //   const auto e = static_cast<int>(end);
    //   const auto c = static_cast<MKL_INT>(count);
    //   viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream_, c, result.data(),
    //   b, e);
    // }
    // #else
    fallback_func(beg, end, result);
    // #endif
    return result;
  }

 private:
  PrefetchedRandint prefetched_;
#if WITH_MKL_BLAS()
  VSLStreamStatePtr stream_;
#endif
};

/**
 * Amortized O(1) uniform random reals within [0,1).
 *
 * Reduced torch random overheads by prefetching large random numbers in
 * advance. Prefetched random reals are consumed on demand.
 *
 */
class PrefetchedRandreal {
 public:
  PrefetchedRandreal() : PrefetchedRandreal(RAND_PREFETCH_SIZE) {}
  PrefetchedRandreal(int size) { prefetch(size); }

  /**
   * Generate next random real number in [0,1) uniformly
   *
   * @tparam T user type of random numbers
   *
   * @returns the chosen number in [0,1)
   */
  template <typename T>
  T next() {
    // Consume some bits
    if (size_ > 0) {
      size_--;
    } else {
      // Prefetch if no enough real numbers
      prefetch(prefetched_randreal_.size(0));
    }

    // torch::rand gives floats by default.
    float* prefetch_ptr = prefetched_randreal_.data_ptr<float>();

    float res = prefetch_ptr[size_];

    // Safe conversion because the result is in range [0,1)
    return (T)res;
  }

 private:
  // Prefetch random real numbers. In-place random if prefetching size is the
  // same. FP32 precision is sufficient for all types of random real numbers.
  void prefetch(int size) {
    if (prefetched_randreal_.size(0) != size) {
      prefetched_randreal_ = at::rand({size});
    } else {
      prefetched_randreal_.uniform_();
    }
    size_ = size - 1;
  }

  at::Tensor prefetched_randreal_;
  int size_;
};

/**
 * Randreal functor for uniform real distribution.
 * Wrapped PrefetchedRandreal as its efficient core implementation.
 */
template <typename T>
class RandrealEngine {
 public:
  RandrealEngine() : prefetched_(RAND_PREFETCH_SIZE) {}

  // Uniform random number within range [beg, end)
  T operator()() { return prefetched_.next<T>(); }

 private:
  PrefetchedRandreal prefetched_;
};

}  // namespace random
}  // namespace pyg
