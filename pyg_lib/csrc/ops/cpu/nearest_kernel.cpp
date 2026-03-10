#include "../nearest.h"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

namespace {

at::Tensor nearest_kernel(const at::Tensor& x,
                          const at::Tensor& y,
                          const std::optional<at::Tensor>& ptr_x,
                          const std::optional<at::Tensor>& ptr_y) {
  auto out = at::empty({x.size(0)}, x.options().dtype(at::kLong));

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "nearest_cpu", [&] {
    auto x_data = x.data_ptr<scalar_t>();
    auto y_data = y.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<int64_t>();
    auto dim = x.size(1);

    if (!ptr_x.has_value()) {
      for (int64_t i = 0; i < x.size(0); i++) {
        scalar_t best_dist = std::numeric_limits<scalar_t>::max();
        int64_t best_idx = 0;
        for (int64_t j = 0; j < y.size(0); j++) {
          scalar_t dist = 0;
          for (int64_t d = 0; d < dim; d++) {
            scalar_t diff = x_data[i * dim + d] - y_data[j * dim + d];
            dist += diff * diff;
          }
          if (dist < best_dist) {
            best_dist = dist;
            best_idx = j;
          }
        }
        out_data[i] = best_idx;
      }
    } else {
      auto ptr_x_data = ptr_x.value().data_ptr<int64_t>();
      auto ptr_y_data = ptr_y.value().data_ptr<int64_t>();
      auto num_batches = ptr_x.value().size(0) - 1;

      for (int64_t b = 0; b < num_batches; b++) {
        auto x_start = ptr_x_data[b], x_end = ptr_x_data[b + 1];
        auto y_start = ptr_y_data[b], y_end = ptr_y_data[b + 1];

        for (int64_t i = x_start; i < x_end; i++) {
          scalar_t best_dist = std::numeric_limits<scalar_t>::max();
          int64_t best_idx = y_start;
          for (int64_t j = y_start; j < y_end; j++) {
            scalar_t dist = 0;
            for (int64_t d = 0; d < dim; d++) {
              scalar_t diff = x_data[i * dim + d] - y_data[j * dim + d];
              dist += diff * diff;
            }
            if (dist < best_dist) {
              best_dist = dist;
              best_idx = j;
            }
          }
          out_data[i] = best_idx;
        }
      }
    }
  });

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::nearest"), TORCH_FN(nearest_kernel));
}

}  // namespace ops
}  // namespace pyg
