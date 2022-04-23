#include "random_walk.h"

namespace pyg {
namespace sampler {

at::Tensor random_walk(const at::Tensor& crow,
                       const at::Tensor& col,
                       const at::Tensor& seed,
                       int64_t walk_length,
                       double p,
                       double q) {
  return seed;
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::ps_roi_align(Tensor input, Tensor rois, float "
      "spatial_scale, int pooled_height, int pooled_width, int sampling_ratio) "
      "-> (Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::_ps_roi_align_backward(Tensor grad, Tensor rois, Tensor "
      "channel_mapping, float spatial_scale, int pooled_height, int "
      "pooled_width, int sampling_ratio, int batch_size, int channels, int "
      "height, int width) -> Tensor"));
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg:random_walk(Tensor crow, Tensor col, Tensor seed, int walk_length, "
      "float p, float q) -> Tensor"))
}

}  // namespace sampler
}  // namespace pyg
