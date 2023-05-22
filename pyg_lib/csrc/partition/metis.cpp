#include "metis.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

#include "pyg_lib/csrc/utils/check.h"

namespace pyg {
namespace partition {

at::Tensor metis(const at::Tensor& rowptr,
                 const at::Tensor& col,
                 int64_t num_partitions,
                 const c10::optional<at::Tensor>& node_weight,
                 const c10::optional<at::Tensor>& edge_weight,
                 bool recursive) {
  at::TensorArg rowptr_t{rowptr, "rowtpr", 1};
  at::TensorArg col_t{col, "col", 1};

  at::CheckedFrom c = "metis";
  at::checkAllDefined(c, {rowptr_t, col_t});
  at::checkAllSameType(c, {rowptr_t, col_t});

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::metis", "")
                       .typed<decltype(metis)>();
  return op.call(rowptr, col, num_partitions, node_weight, edge_weight,
                 recursive);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::metis(Tensor rowptr, Tensor col, int num_partitions, Tensor? "
      "node_weight = None, Tensor? edge_weight = None, bool recursive = False) "
      "-> Tensor"));
}

}  // namespace partition
}  // namespace pyg
