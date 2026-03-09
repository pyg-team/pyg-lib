#include "cluster.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

PYG_API at::Tensor grid_cluster(const at::Tensor& pos,
                                const at::Tensor& size,
                                const std::optional<at::Tensor>& start,
                                const std::optional<at::Tensor>& end) {
  at::TensorArg pos_arg{pos, "pos", 0};
  at::TensorArg size_arg{size, "size", 1};
  at::CheckedFrom c{"grid_cluster"};

  at::checkAllDefined(c, {pos_arg, size_arg});

  auto pos_2d = pos.view({pos.size(0), -1}).contiguous();
  auto size_c = size.contiguous();

  TORCH_CHECK(size_c.numel() == pos_2d.size(1),
              "size.numel() must equal pos dimension count");

  if (start.has_value()) {
    TORCH_CHECK(start.value().numel() == pos_2d.size(1),
                "start.numel() must equal pos dimension count");
  }
  if (end.has_value()) {
    TORCH_CHECK(end.value().numel() == pos_2d.size(1),
                "end.numel() must equal pos dimension count");
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::grid_cluster", "")
                       .typed<decltype(grid_cluster)>();
  return op.call(pos_2d, size_c, start, end);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::grid_cluster(Tensor pos, Tensor size, "
      "Tensor? start=None, Tensor? end=None) -> Tensor"));
}

}  // namespace ops
}  // namespace pyg
