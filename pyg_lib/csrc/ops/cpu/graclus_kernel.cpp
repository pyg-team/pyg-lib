#include "../graclus.h"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

namespace {

at::Tensor graclus_kernel(const at::Tensor& rowptr,
                          const at::Tensor& col,
                          const std::optional<at::Tensor>& weight) {
  int64_t num_nodes = rowptr.numel() - 1;
  auto out = at::full({num_nodes}, -1, rowptr.options());
  auto node_perm = at::randperm(num_nodes, rowptr.options());

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto node_perm_data = node_perm.data_ptr<int64_t>();
  auto out_data = out.data_ptr<int64_t>();

  if (!weight.has_value()) {
    for (int64_t n = 0; n < num_nodes; n++) {
      auto u = node_perm_data[n];

      if (out_data[u] >= 0)
        continue;

      out_data[u] = u;

      int64_t row_start = rowptr_data[u], row_end = rowptr_data[u + 1];

      for (int64_t e = 0; e < row_end - row_start; e++) {
        auto v = col_data[row_start + e];

        if (out_data[v] >= 0)
          continue;

        out_data[u] = std::min(u, v);
        out_data[v] = std::min(u, v);
        break;
      }
    }
  } else {
    auto scalar_type = weight.value().scalar_type();
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type,
        "graclus_cpu", [&] {
          auto weight_data = weight.value().data_ptr<scalar_t>();

          for (int64_t n = 0; n < num_nodes; n++) {
            auto u = node_perm_data[n];

            if (out_data[u] >= 0)
              continue;

            auto v_max = u;
            scalar_t w_max = (scalar_t)0.;

            for (int64_t e = rowptr_data[u]; e < rowptr_data[u + 1]; e++) {
              auto v = col_data[e];

              if (out_data[v] >= 0)
                continue;

              if (weight_data[e] >= w_max) {
                v_max = v;
                w_max = weight_data[e];
              }
            }

            out_data[u] = std::min(u, v_max);
            out_data[v_max] = std::min(u, v_max);
          }
        });
  }

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::graclus_cluster"),
         TORCH_FN(graclus_kernel));
}

}  // namespace ops
}  // namespace pyg
