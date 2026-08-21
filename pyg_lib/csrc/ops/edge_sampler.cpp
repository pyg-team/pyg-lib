#include "edge_sampler.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

PYG_API at::Tensor edge_sample(const at::Tensor& start,
                               const at::Tensor& rowptr,
                               int64_t count,
                               double factor) {
  at::TensorArg start_arg{start, "start", 0};
  at::TensorArg rowptr_arg{rowptr, "rowptr", 1};
  at::CheckedFrom c{"edge_sample"};

  at::checkAllDefined(c, {start_arg, rowptr_arg});
  at::checkDim(c, start_arg, 1);
  at::checkDim(c, rowptr_arg, 1);

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::edge_sample", "")
                       .typed<decltype(edge_sample)>();
  return op.call(start, rowptr, count, factor);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(
      TORCH_SELECTIVE_SCHEMA("pyg::edge_sample(Tensor start, Tensor rowptr, "
                             "int count=0, float factor=1.0) -> Tensor"));
}

}  // namespace ops
}  // namespace pyg
