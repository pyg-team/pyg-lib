#include <ATen/ATen.h>
#include <torch/library.h>

#include "pyg_lib/csrc/utils/cpu/convert.h"

namespace pyg {
namespace sampler {

namespace {

std::tuple<at::Tensor, at::Tensor, at::Tensor> subgraph_kernel(
    const at::Tensor& rowptr,
    const at::Tensor& col,
    const at::Tensor& nodes) {
  TORCH_CHECK(rowptr.is_cpu(), "'rowptr' must be a CPU tensor");
  TORCH_CHECK(col.is_cpu(), "'col' must be a CPU tensor");
  TORCH_CHECK(nodes.is_cpu(), "'nodes' must be a CPU tensor");

  const auto out_rowptr = rowptr.new_empty({nodes.size(0) + 1});
  at::Tensor out_col, out_edge_id;

  AT_DISPATCH_INTEGRAL_TYPES(nodes.scalar_type(), "subgraph_kernel", [&] {
    const auto rowptr_data = rowptr.data_ptr<scalar_t>();
    const auto col_data = col.data_ptr<scalar_t>();
    const auto nodes_data = nodes.data_ptr<scalar_t>();

    auto out_rowptr_data = out_rowptr.data_ptr<scalar_t>();
    out_rowptr_data[0] = 0;

    std::unordered_map<scalar_t, scalar_t> to_local_node;
    for (scalar_t i = 0; i < nodes.size(0); ++i)
      to_local_node.insert({nodes_data[i], i});

    scalar_t offset = 0;
    std::vector<scalar_t> out_col_vec, out_edge_id_vec;
    for (scalar_t i = 0; i < nodes.size(0); ++i) {
      const auto v = nodes_data[i];
      for (scalar_t j = rowptr_data[v]; j < rowptr_data[v + 1]; ++j) {
        const auto w = col_data[j];
        const auto search = to_local_node.find(w);
        if (search != to_local_node.end()) {
          out_col_vec.push_back(search->second);
          out_edge_id_vec.push_back(j);
          offset++;
        }
      }
      out_rowptr_data[i + 1] = offset;
    }
    out_col = pyg::utils::from_vector(out_col_vec);
    out_edge_id = pyg::utils::from_vector(out_edge_id_vec);
  });

  return std::make_tuple(out_rowptr, out_col, out_edge_id);
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::subgraph"), TORCH_FN(subgraph_kernel));
}

}  // namespace sampler
}  // namespace pyg
