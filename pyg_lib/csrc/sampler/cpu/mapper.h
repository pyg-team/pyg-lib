#pragma once

#include <ATen/ATen.h>

namespace pyg {
namespace sampler {

// TODO Implement `Mapper` as an interface/abstract class to allow for other
// implementations as well.
template <typename scalar_t>
class Mapper {
 public:
  Mapper(scalar_t num_nodes, scalar_t num_entries)
      : num_nodes(num_nodes), num_entries(num_entries) {
    // Use a some simple heuristic to determine whether we can use a std::vector
    // to perform the mapping instead of relying on the more memory-friendly,
    // but slower std::unordered_map implementation:
    use_vec = (num_nodes < 1000000) || (num_entries > num_nodes / 10);

    if (use_vec)
      to_local_vec = std::vector<scalar_t>(num_nodes, -1);
  }

  void fill(const scalar_t* nodes_data, const scalar_t size) {
    if (use_vec) {
      for (scalar_t i = 0; i < size; ++i) {
        to_local_vec[nodes_data[i]] = i;
      }
    } else {
      for (scalar_t i = 0; i < size; ++i) {
        to_local_map.insert({nodes_data[i], i});
      }
    }
  }

  void fill(const at::Tensor& nodes) {
    fill(nodes.data_ptr<scalar_t>(), nodes.numel());
  }

  bool exists(const scalar_t& node) const {
    if (use_vec)
      return to_local_vec[node] >= 0;
    else
      return to_local_map.count(node) > 0;
  }

  scalar_t map(const scalar_t& node) const {
    if (use_vec)
      return to_local_vec[node];
    else {
      const auto search = to_local_map.find(node);
      return search != to_local_map.end() ? search->second : -1;
    }
  }

 private:
  scalar_t num_nodes, num_entries;

  bool use_vec;
  std::vector<scalar_t> to_local_vec;
  std::unordered_map<scalar_t, scalar_t> to_local_map;
};

}  // namespace sampler
}  // namespace pyg
