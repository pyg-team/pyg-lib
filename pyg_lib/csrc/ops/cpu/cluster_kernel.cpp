#include "../cluster.h"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

namespace {

at::Tensor grid_cluster_kernel(const at::Tensor& pos,
                               const at::Tensor& size,
                               const std::optional<at::Tensor>& optional_start,
                               const std::optional<at::Tensor>& optional_end) {
  auto N = pos.size(0);
  auto D = pos.size(1);

  at::Tensor start;
  if (optional_start.has_value())
    start = optional_start.value().contiguous();
  else
    start = std::get<0>(pos.min(0));

  at::Tensor end;
  if (optional_end.has_value())
    end = optional_end.value().contiguous();
  else
    end = std::get<0>(pos.max(0));

  auto pos_shifted = pos - start.unsqueeze(0);

  auto num_voxels =
      (end - start).div(size, /*rounding_mode=*/"trunc").to(at::kLong) + 1;
  num_voxels = num_voxels.cumprod(0);
  num_voxels = at::cat({at::ones({1}, num_voxels.options()), num_voxels}, 0);
  num_voxels = num_voxels.narrow(0, 0, D);

  auto out = pos_shifted.div(size.view({1, -1}), /*rounding_mode=*/"trunc")
                 .to(at::kLong);
  out *= num_voxels.view({1, -1});
  out = out.sum(1);

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::grid_cluster"),
         TORCH_FN(grid_cluster_kernel));
}

}  // namespace ops
}  // namespace pyg
