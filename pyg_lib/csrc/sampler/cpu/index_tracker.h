#pragma once

#include <ATen/ATen.h>

#include "parallel_hashmap/phmap.h"

namespace pyg {
namespace sampler {

template <typename scalar_t>
class IndexTracker {
 public:
  IndexTracker(const size_t& size) : size(size) {
    // TODO: add better switching threshold value mechanism?
    use_vec = (size < 100000);

    if (use_vec)
      vec.resize(size, 0);
  }

  bool try_insert(const scalar_t& index) {
    if (use_vec) {
      if (vec[index] == 0) {
        vec[index] = 1;
        return true;
      } else {
        return false;
      }
    } else {
      return set.insert(index).second;
    }
  }

  void insert(const scalar_t& index) {
    if (use_vec) {
      vec[index] = 1;
    } else {
      set.insert(index);
    }
  }

 private:
  const size_t& size;

  bool use_vec;
  std::vector<char> vec;
  phmap::flat_hash_set<scalar_t> set;
};

}  // namespace sampler
}  // namespace pyg
