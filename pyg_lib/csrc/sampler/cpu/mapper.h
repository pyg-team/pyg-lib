#pragma once

#include <ATen/ATen.h>

#include "parallel_hashmap/phmap.h"

namespace pyg {
namespace sampler {

// TODO Implement `Mapper` as an interface/abstract class to allow for other
// implementations as well.
template <typename node_t, typename scalar_t>
class Mapper {
 public:
  Mapper(const size_t num_nodes, const size_t num_entries = -1)
      : num_nodes(num_nodes), num_entries(num_entries) {
    // We use some simple heuristic to determine whether we can use a vector
    // to perform the mapping instead of relying on the more memory-friendly,
    // but slower hash map implementation. As a general rule of thumb, we are
    // safe to use vectors in case the number of nodes are small, or it is
    // expected that we sample a large amount of nodes.
    use_vec = (num_nodes < 1000000) || (num_entries > num_nodes / 10);

    if (num_nodes <= 0) {  // == `num_nodes` is undefined:
      use_vec = false;
    }

    // We can only utilize vector mappings in case entries are scalar:
    if (!std::is_scalar<node_t>::value) {
      use_vec = false;
    }

    if (use_vec) {
      to_local_vec.resize(num_nodes, -1);
    }
  }

  std::pair<scalar_t, bool> insert(const node_t& node) {
    std::pair<scalar_t, bool> res;
    if (use_vec) {
      if constexpr (std::is_scalar<node_t>::value) {
        auto old = to_local_vec[node];
        res = std::pair<scalar_t, bool>(old == -1 ? curr : old, old == -1);
        if (res.second)
          to_local_vec[node] = curr;
      }
    } else {
      auto out = to_local_map.insert({node, curr});
      res = std::pair<scalar_t, bool>(out.first->second, out.second);
    }
    if (res.second) {
      curr++;
    }
    return res;
  }

  void fill(const node_t* nodes, const size_t size) {
    for (size_t i = 0; i < size; ++i) {
      insert(nodes[i]);
    }
  }

  void fill(const at::Tensor& nodes) {
    fill(nodes.data_ptr<node_t>(), nodes.numel());
  }

  bool exists(const node_t& node) {
    if (use_vec) {
      return to_local_vec[node] >= 0;
    } else {
      return to_local_map.count(node) > 0;
    }
  }

  scalar_t map(const node_t& node) {
    if (use_vec) {
      return to_local_vec[node];
    } else {
      const auto search = to_local_map.find(node);
      return search != to_local_map.end() ? search->second : -1;
    }
  }

 private:
  const size_t num_nodes, num_entries;
  scalar_t curr = 0;

  bool use_vec;
  std::vector<scalar_t> to_local_vec;
  phmap::flat_hash_map<node_t, scalar_t> to_local_map;
};

}  // namespace sampler
}  // namespace pyg
