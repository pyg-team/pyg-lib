#include "scatter.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

PYG_API at::Tensor scatter_sum(const at::Tensor& src,
                               const at::Tensor& index,
                               int64_t dim,
                               const std::optional<at::Tensor>& out,
                               std::optional<int64_t> dim_size) {
  at::TensorArg src_arg{src, "src", 0};
  at::TensorArg index_arg{index, "index", 1};
  at::CheckedFrom c{"scatter_sum"};

  at::checkAllDefined(c, {src_arg, index_arg});
  TORCH_CHECK(src.device() == index.device(),
              "scatter_sum: src and index must be on the same device "
              "(got src=",
              src.device(), ", index=", index.device(), ")");
  if (out.has_value()) {
    at::TensorArg out_arg{out.value(), "out", 3};
    at::checkAllDefined(c, {out_arg});
    TORCH_CHECK(src.device() == out.value().device(),
                "scatter_sum: src and out must be on the same device "
                "(got src=",
                src.device(), ", out=", out.value().device(), ")");
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::scatter_sum", "")
                       .typed<decltype(scatter_sum)>();
  return op.call(src, index, dim, out, dim_size);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::scatter_sum(Tensor src, Tensor index, int dim=-1, "
      "Tensor? out=None, int? dim_size=None) -> Tensor"));
}

}  // namespace ops
}  // namespace pyg
