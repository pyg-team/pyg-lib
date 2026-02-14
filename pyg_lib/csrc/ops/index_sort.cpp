#include "index_sort.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

PYG_API std::tuple<at::Tensor, at::Tensor> index_sort(
    const at::Tensor& input,
    const at::optional<int64_t> max) {
  at::TensorArg input_arg{input, "input", 0};
  at::CheckedFrom c{"index_sort"};

  at::checkAllDefined(c, {input_arg});
  at::checkContiguous(c, input_arg);

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::index_sort", "")
                       .typed<decltype(index_sort)>();

  return op.call(input, max);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::index_sort(Tensor indices, int? max = None) -> (Tensor, Tensor)"));
}

}  // namespace ops
}  // namespace pyg
