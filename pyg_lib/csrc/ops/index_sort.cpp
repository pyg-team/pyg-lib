#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::index_sort(Tensor indices, int? max = None) -> (Tensor, Tensor)"));
}

} // namespace ops
} // namespace pyg
