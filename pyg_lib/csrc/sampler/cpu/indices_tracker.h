#pragma once

#include <ATen/ATen.h>

#include "parallel_hashmap/phmap.h"

namespace pyg {
namespace sampler {

template <typename scalar_t>
class IndicesTracker {
 public:
  IndicesTracker(const scalar_t& size) : size(size) {
    // TODO: add better switching threshold value mechanism?
    use_vec = (size < 100000);
    if (use_vec)
      rnd_indices_vec.resize(size, 0);
  }

  bool tryInsert(const scalar_t& index) {
    if (use_vec) {
      if (rnd_indices_vec[index] == 0) {
        rnd_indices_vec[index] = 1;
        return true;
      } else {
        return false;
      }
    } else {
      return rnd_indices_set.insert(index).second;
    }
  }

  void insert(const scalar_t& index) {
    if (use_vec) {
      rnd_indices_vec[index] = 1;
    } else {
      rnd_indices_set.insert(index);
    }
  }

 private:
  const scalar_t& size;
  bool use_vec;
  std::vector<char> rnd_indices_vec;
  phmap::flat_hash_set<scalar_t> rnd_indices_set;
};

}  // namespace sampler
}  // namespace pyg
