#include "graclus.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

PYG_API at::Tensor graclus_cluster(const at::Tensor& rowptr,
                                   const at::Tensor& col,
                                   const std::optional<at::Tensor>& weight) {
  at::TensorArg rowptr_arg{rowptr, "rowptr", 0};
  at::TensorArg col_arg{col, "col", 1};
  at::CheckedFrom c{"graclus_cluster"};

  at::checkAllDefined(c, {rowptr_arg, col_arg});
  at::checkDim(c, rowptr_arg, 1);
  at::checkDim(c, col_arg, 1);

  if (weight.has_value()) {
    TORCH_CHECK(weight.value().dim() == 1, "weight must be 1-dimensional");
    TORCH_CHECK(weight.value().numel() == col.numel(),
                "weight must have the same number of elements as col");
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::graclus_cluster", "")
                       .typed<decltype(graclus_cluster)>();
  return op.call(rowptr, col, weight);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(
      TORCH_SELECTIVE_SCHEMA("pyg::graclus_cluster(Tensor rowptr, Tensor col, "
                             "Tensor? weight=None) -> Tensor"));
}

}  // namespace ops
}  // namespace pyg
