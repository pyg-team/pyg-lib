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

PYG_API at::Tensor scatter_mul(const at::Tensor& src,
                               const at::Tensor& index,
                               int64_t dim,
                               const std::optional<at::Tensor>& out,
                               std::optional<int64_t> dim_size) {
  at::TensorArg src_arg{src, "src", 0};
  at::TensorArg index_arg{index, "index", 1};
  at::CheckedFrom c{"scatter_mul"};

  at::checkAllDefined(c, {src_arg, index_arg});
  TORCH_CHECK(src.device() == index.device(),
              "scatter_mul: src and index must be on the same device "
              "(got src=",
              src.device(), ", index=", index.device(), ")");
  if (out.has_value()) {
    at::TensorArg out_arg{out.value(), "out", 3};
    at::checkAllDefined(c, {out_arg});
    TORCH_CHECK(src.device() == out.value().device(),
                "scatter_mul: src and out must be on the same device "
                "(got src=",
                src.device(), ", out=", out.value().device(), ")");
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::scatter_mul", "")
                       .typed<decltype(scatter_mul)>();
  return op.call(src, index, dim, out, dim_size);
}

PYG_API at::Tensor scatter_mean(const at::Tensor& src,
                                const at::Tensor& index,
                                int64_t dim,
                                const std::optional<at::Tensor>& out,
                                std::optional<int64_t> dim_size) {
  at::TensorArg src_arg{src, "src", 0};
  at::TensorArg index_arg{index, "index", 1};
  at::CheckedFrom c{"scatter_mean"};

  at::checkAllDefined(c, {src_arg, index_arg});
  TORCH_CHECK(src.device() == index.device(),
              "scatter_mean: src and index must be on the same device "
              "(got src=",
              src.device(), ", index=", index.device(), ")");
  if (out.has_value()) {
    at::TensorArg out_arg{out.value(), "out", 3};
    at::checkAllDefined(c, {out_arg});
    TORCH_CHECK(src.device() == out.value().device(),
                "scatter_mean: src and out must be on the same device "
                "(got src=",
                src.device(), ", out=", out.value().device(), ")");
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::scatter_mean", "")
                       .typed<decltype(scatter_mean)>();
  return op.call(src, index, dim, out, dim_size);
}

PYG_API std::tuple<at::Tensor, at::Tensor> scatter_min(
    const at::Tensor& src,
    const at::Tensor& index,
    int64_t dim,
    const std::optional<at::Tensor>& out,
    std::optional<int64_t> dim_size) {
  at::TensorArg src_arg{src, "src", 0};
  at::TensorArg index_arg{index, "index", 1};
  at::CheckedFrom c{"scatter_min"};

  at::checkAllDefined(c, {src_arg, index_arg});
  TORCH_CHECK(src.device() == index.device(),
              "scatter_min: src and index must be on the same device "
              "(got src=",
              src.device(), ", index=", index.device(), ")");
  if (out.has_value()) {
    at::TensorArg out_arg{out.value(), "out", 3};
    at::checkAllDefined(c, {out_arg});
    TORCH_CHECK(src.device() == out.value().device(),
                "scatter_min: src and out must be on the same device "
                "(got src=",
                src.device(), ", out=", out.value().device(), ")");
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::scatter_min", "")
                       .typed<decltype(scatter_min)>();
  return op.call(src, index, dim, out, dim_size);
}

PYG_API std::tuple<at::Tensor, at::Tensor> scatter_max(
    const at::Tensor& src,
    const at::Tensor& index,
    int64_t dim,
    const std::optional<at::Tensor>& out,
    std::optional<int64_t> dim_size) {
  at::TensorArg src_arg{src, "src", 0};
  at::TensorArg index_arg{index, "index", 1};
  at::CheckedFrom c{"scatter_max"};

  at::checkAllDefined(c, {src_arg, index_arg});
  TORCH_CHECK(src.device() == index.device(),
              "scatter_max: src and index must be on the same device "
              "(got src=",
              src.device(), ", index=", index.device(), ")");
  if (out.has_value()) {
    at::TensorArg out_arg{out.value(), "out", 3};
    at::checkAllDefined(c, {out_arg});
    TORCH_CHECK(src.device() == out.value().device(),
                "scatter_max: src and out must be on the same device "
                "(got src=",
                src.device(), ", out=", out.value().device(), ")");
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::scatter_max", "")
                       .typed<decltype(scatter_max)>();
  return op.call(src, index, dim, out, dim_size);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::scatter_sum(Tensor src, Tensor index, int dim=-1, "
      "Tensor? out=None, int? dim_size=None) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::scatter_mul(Tensor src, Tensor index, int dim=-1, "
      "Tensor? out=None, int? dim_size=None) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::scatter_mean(Tensor src, Tensor index, int dim=-1, "
      "Tensor? out=None, int? dim_size=None) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::scatter_min(Tensor src, Tensor index, int dim=-1, "
      "Tensor? out=None, int? dim_size=None) -> (Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::scatter_max(Tensor src, Tensor index, int dim=-1, "
      "Tensor? out=None, int? dim_size=None) -> (Tensor, Tensor)"));
}

}  // namespace ops
}  // namespace pyg
