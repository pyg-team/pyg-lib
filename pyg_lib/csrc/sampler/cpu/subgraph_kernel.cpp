#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "pyg_lib/csrc/utils/cpu/convert.h"

namespace pyg {
namespace sampler {

namespace {

template <typename scalar_t>
class Mapper {
 public:
  Mapper(scalar_t num_nodes, scalar_t num_entries)
      : num_nodes(num_nodes), num_entries(num_entries) {
    // Use a some simple heuristic to determine whether we can use a std::vector
    // to perform the mapping instead of relying on the more memory-friendly,
    // but slower std::unordered_map:
    use_vec = (num_nodes < 1000000) || (num_entries > num_nodes / 10);

    if (use_vec)
      to_local_vec = std::vector<scalar_t>(num_nodes, -1);
  }

  void fill(const scalar_t* nodes_data, const scalar_t size) {
    if (use_vec) {
      for (scalar_t i = 0; i < size; ++i)
        to_local_vec[nodes_data[i]] = i;
    } else {
      for (scalar_t i = 0; i < size; ++i)
        to_local_map.insert({nodes_data[i], i});
    }
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
  scalar_t num_nodes, num_entries;

  bool use_vec;
  std::vector<scalar_t> to_local_vec;
  std::unordered_map<scalar_t, scalar_t> to_local_map;
};

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> subgraph_kernel(
    const at::Tensor& rowptr,
    const at::Tensor& col,
    const at::Tensor& nodes,
    const bool return_edge_id) {
  TORCH_CHECK(rowptr.is_cpu(), "'rowptr' must be a CPU tensor");
  TORCH_CHECK(col.is_cpu(), "'col' must be a CPU tensor");
  TORCH_CHECK(nodes.is_cpu(), "'nodes' must be a CPU tensor");

  const auto out_rowptr = rowptr.new_empty({nodes.size(0) + 1});
  at::Tensor out_col;
  c10::optional<at::Tensor> out_edge_id = c10::nullopt;

  AT_DISPATCH_INTEGRAL_TYPES(nodes.scalar_type(), "subgraph_kernel", [&] {
    auto mapper = Mapper<scalar_t>(rowptr.size(0) - 1, nodes.size(0));
    mapper.fill(nodes);

    const auto rowptr_data = rowptr.data_ptr<scalar_t>();
    const auto col_data = col.data_ptr<scalar_t>();
    const auto nodes_data = nodes.data_ptr<scalar_t>();

    // We first iterate over all nodes and collect information about the number
    // of edges in the induced subgraph.
    const auto deg = rowptr.new_empty({nodes.size(0)});
    auto deg_data = deg.data_ptr<scalar_t>();
    auto grain_size = at::internal::GRAIN_SIZE;
    at::parallel_for(0, nodes.size(0), grain_size, [&](int64_t _s, int64_t _e) {
      for (size_t i = _s; i < _e; ++i) {
        const auto v = nodes_data[i];
        // Iterate over all neighbors and check if they are part of `nodes`:
        scalar_t d = 0;
        for (size_t j = rowptr_data[v]; j < rowptr_data[v + 1]; ++j) {
          if (mapper.exists(col_data[j]))
            d++;
        }
        deg_data[i] = d;
      }
    });

    auto out_rowptr_data = out_rowptr.data_ptr<scalar_t>();
    out_rowptr_data[0] = 0;
    auto tmp = out_rowptr.narrow(0, 1, nodes.size(0));
    at::cumsum_out(tmp, deg, /*dim=*/0);

    out_col = col.new_empty({out_rowptr_data[nodes.size(0)]});
    auto out_col_data = out_col.data_ptr<scalar_t>();
    scalar_t* out_edge_id_data;
    if (return_edge_id) {
      out_edge_id = col.new_empty({out_rowptr_data[nodes.size(0)]});
      out_edge_id_data = out_edge_id.value().data_ptr<scalar_t>();
    }

    // Customize `grain_size` based on the work each thread does (it will need
    // to find `col.size(0) / nodes.size(0)` neighbors on average).
    // TODO Benchmark this customization
    grain_size = std::max<int64_t>(out_col.size(0) / nodes.size(0), 1);
    grain_size = at::internal::GRAIN_SIZE / grain_size;
    at::parallel_for(0, nodes.size(0), grain_size, [&](int64_t _s, int64_t _e) {
      for (scalar_t i = _s; i < _e; ++i) {
        const auto v = nodes_data[i];
        // Iterate over all neighbors and check if they are part of `nodes`:
        scalar_t offset = out_rowptr_data[i];
        for (scalar_t j = rowptr_data[v]; j < rowptr_data[v + 1]; ++j) {
          const auto w = mapper.map(col_data[j]);
          if (w >= 0) {
            out_col_data[offset] = w;
            if (return_edge_id)
              out_edge_id_data[offset] = j;
          }
        }
      }
    });
  });

  return std::make_tuple(out_rowptr, out_col, out_edge_id);
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::subgraph"), TORCH_FN(subgraph_kernel));
}

}  // namespace sampler
}  // namespace pyg
