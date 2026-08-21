#include "../fps.h"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

namespace {

at::Tensor fps_kernel(const at::Tensor& src,
                      const at::Tensor& ptr,
                      double ratio,
                      bool random_start) {
  auto N = src.size(0);
  auto D = src.size(1);
  auto batch_size = ptr.numel() - 1;

  auto deg = ptr.narrow(0, 1, batch_size) - ptr.narrow(0, 0, batch_size);
  auto out_ptr = deg.to(at::kFloat) * ratio;
  out_ptr = out_ptr.ceil().to(at::kLong).cumsum(0);

  auto out = at::empty({out_ptr[-1].data_ptr<int64_t>()[0]}, ptr.options());

  auto ptr_data = ptr.data_ptr<int64_t>();
  auto out_ptr_data = out_ptr.data_ptr<int64_t>();
  auto out_data = out.data_ptr<int64_t>();

  int64_t grain_size = 1;
  at::parallel_for(0, batch_size, grain_size, [&](int64_t begin, int64_t end) {
    for (int64_t b = begin; b < end; b++) {
      auto src_start = ptr_data[b];
      auto src_end = ptr_data[b + 1];
      auto out_start = b == 0 ? 0 : out_ptr_data[b - 1];
      auto out_end = out_ptr_data[b];

      auto y = src.narrow(0, src_start, src_end - src_start);

      int64_t start_idx = 0;
      if (random_start)
        start_idx = rand() % y.size(0);

      out_data[out_start] = src_start + start_idx;
      auto dist = (y - y[start_idx]).pow_(2).sum(1);

      for (int64_t i = 1; i < out_end - out_start; i++) {
        int64_t argmax = dist.argmax().data_ptr<int64_t>()[0];
        out_data[out_start + i] = src_start + argmax;
        dist = at::min(dist, (y - y[argmax]).pow_(2).sum(1));
      }
    }
  });

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::fps"), TORCH_FN(fps_kernel));
}

}  // namespace ops
}  // namespace pyg
