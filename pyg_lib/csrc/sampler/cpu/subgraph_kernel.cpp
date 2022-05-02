#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "pyg_lib/csrc/utils/cpu/convert.h"

namespace pyg {
namespace sampler {

namespace {

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> subgraph_kernel(
    const at::Tensor& rowptr,
    const at::Tensor& col,
    const at::Tensor& nodes,
    const bool return_edge_id) {
  TORCH_CHECK(rowptr.is_cpu(), "'rowptr' must be a CPU tensor");
  TORCH_CHECK(col.is_cpu(), "'col' must be a CPU tensor");
  TORCH_CHECK(nodes.is_cpu(), "'nodes' must be a CPU tensor");

  const auto deg = rowptr.new_empty({nodes.size(0)});
  const auto out_rowptr = rowptr.new_empty({nodes.size(0) + 1});
  at::Tensor out_col;
  c10::optional<at::Tensor> out_edge_id = c10::nullopt;

  AT_DISPATCH_INTEGRAL_TYPES(nodes.scalar_type(), "subgraph_kernel", [&] {
    const auto rowptr_data = rowptr.data_ptr<scalar_t>();
    const auto col_data = col.data_ptr<scalar_t>();
    const auto nodes_data = nodes.data_ptr<scalar_t>();

    std::unordered_map<scalar_t, scalar_t> to_local_node;
    for (scalar_t i = 0; i < nodes.size(0); ++i)  // TODO parallelize
      to_local_node.insert({nodes_data[i], i});

    // We first iterate over all nodes and collect information about the number
    // of edges in the induced subgraph.
    auto deg_data = deg.data_ptr<scalar_t>();
    auto grain_size = at::internal::GRAIN_SIZE;
    at::parallel_for(0, nodes.size(0), grain_size, [&](int64_t _s, int64_t _e) {
      for (scalar_t i = _s; i < _e; ++i) {
        const auto v = nodes_data[i];
        // Iterate over all neighbors and check if they are part of `nodes`:
        scalar_t d = 0;
        for (scalar_t j = rowptr_data[v]; j < rowptr_data[v + 1]; ++j) {
          if (to_local_node.count(col_data[j]) > 0)
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
          const auto w = col_data[j];
          const auto search = to_local_node.find(w);
          if (search != to_local_node.end()) {
            out_col_data[offset] = search->second;
            if (return_edge_id)
              out_edge_id_data[offset] = j;
            offset++;
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
