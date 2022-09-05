#pragma once

#include <ATen/ATen.h>

#include "parallel_hashmap/phmap.h"

namespace pyg {
namespace sampler {

// TODO Implement `Mapper` as an interface/abstract class to allow for other
// implementations as well.
template <typename scalar_t>
class Mapper {
 public:
  Mapper(scalar_t num_nodes, scalar_t num_entries)
      : num_nodes(num_nodes), num_entries(num_entries) {
    // We use some simple heuristic to determine whether we can use a vector
    // to perform the mapping instead of relying on the more memory-friendly,
    // but slower hash map implementation. As a general rule of thumb, we are
    // safe to use vectors in case the number of nodes are small, or it is
    // expected that we sample a large amount of nodes.
    use_vec = (num_nodes < 1000000) || (num_entries > num_nodes / 10);

    if (use_vec)
      to_local_vec = std::vector<scalar_t>(num_nodes, -1);
  }

  std::pair<scalar_t, bool> insert(const scalar_t& node) {
    std::pair<scalar_t, bool> res;
    if (use_vec) {
      auto old = to_local_vec[node];
      res = std::pair<scalar_t, bool>(old == -1 ? curr : old, old == -1);
      if (res.second)
        to_local_vec[node] = curr;
    } else {
      auto out = to_local_map.insert({node, curr});
      res = std::pair<scalar_t, bool>(out.first->second, out.second);
    }
    if (res.second)
      curr++;
    return res;
  }

  void fill(const scalar_t* nodes_data, const scalar_t size) {
    for (size_t i = 0; i < size; ++i)
      insert(nodes_data[i]);
  }

  void fill(const at::Tensor& nodes) {
    fill(nodes.data_ptr<scalar_t>(), nodes.numel());
  }

  bool exists(const scalar_t& node) {
    if (use_vec)
      return to_local_vec[node] >= 0;
    else
      return to_local_map.count(node) > 0;
  }

  scalar_t map(const scalar_t& node) {
    if (use_vec)
      return to_local_vec[node];
    else {
      const auto search = to_local_map.find(node);
      return search != to_local_map.end() ? search->second : -1;
    }
  }

 private:
  scalar_t num_nodes, num_entries, curr = 0;

  bool use_vec;
  std::vector<scalar_t> to_local_vec;
  phmap::flat_hash_map<scalar_t, scalar_t> to_local_map;
};

}  // namespace sampler
}  // namespace pyg
