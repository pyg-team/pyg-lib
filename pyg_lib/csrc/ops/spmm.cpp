#include "spmm.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

namespace {

void check_spmm_inputs(const char* name,
                       const at::Tensor& rowptr,
                       const at::Tensor& col,
                       const std::optional<at::Tensor>& value,
                       const at::Tensor& mat) {
  at::TensorArg rowptr_arg{rowptr, "rowptr", 0};
  at::TensorArg col_arg{col, "col", 1};
  at::TensorArg mat_arg{mat, "mat", 3};
  at::CheckedFrom c{name};

  at::checkAllDefined(c, {rowptr_arg, col_arg, mat_arg});
  TORCH_CHECK(rowptr.device() == col.device(), name,
              ": rowptr and col must be on the same device (got rowptr=",
              rowptr.device(), ", col=", col.device(), ")");
  TORCH_CHECK(rowptr.device() == mat.device(), name,
              ": rowptr and mat must be on the same device (got rowptr=",
              rowptr.device(), ", mat=", mat.device(), ")");
  if (value.has_value()) {
    at::TensorArg value_arg{value.value(), "value", 2};
    at::checkAllDefined(c, {value_arg});
    TORCH_CHECK(rowptr.device() == value.value().device(), name,
                ": rowptr and value must be on the same device (got rowptr=",
                rowptr.device(), ", value=", value.value().device(), ")");
  }

  TORCH_CHECK(rowptr.scalar_type() == at::kLong, name,
              ": rowptr must have dtype torch.long");
  TORCH_CHECK(col.scalar_type() == at::kLong, name,
              ": col must have dtype torch.long");
  TORCH_CHECK(rowptr.dim() == 1, name, ": rowptr must be one-dimensional");
  TORCH_CHECK(col.dim() == 1, name, ": col must be one-dimensional");
  TORCH_CHECK(rowptr.numel() > 0, name,
              ": rowptr must contain at least one element");
  TORCH_CHECK(mat.dim() >= 2, name, ": mat must be at least two-dimensional");
  if (value.has_value()) {
    TORCH_CHECK(value.value().dim() == 1, name,
                ": value must be one-dimensional");
    TORCH_CHECK(value.value().size(0) == col.size(0), name,
                ": value.size(0) must match col.size(0)");
    TORCH_CHECK(value.value().scalar_type() == mat.scalar_type(), name,
                ": value and mat must have the same dtype");
  }
}

}  // namespace

PYG_API at::Tensor spmm_sum(const at::Tensor& rowptr,
                            const at::Tensor& col,
                            const std::optional<at::Tensor>& value,
                            const at::Tensor& mat) {
  check_spmm_inputs("spmm_sum", rowptr, col, value, mat);

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::spmm_sum", "")
                       .typed<decltype(spmm_sum)>();
  return op.call(rowptr, col, value, mat);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(
      TORCH_SELECTIVE_SCHEMA("pyg::spmm_sum(Tensor rowptr, Tensor col, "
                             "Tensor? value, Tensor mat) -> Tensor"));
}

}  // namespace ops
}  // namespace pyg
