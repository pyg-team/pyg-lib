#include "segment_coo.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

PYG_API at::Tensor segment_sum_coo(const at::Tensor& src,
                                   const at::Tensor& index,
                                   const std::optional<at::Tensor>& out,
                                   std::optional<int64_t> dim_size) {
  at::TensorArg src_arg{src, "src", 0};
  at::TensorArg index_arg{index, "index", 1};
  at::CheckedFrom c{"segment_sum_coo"};

  at::checkAllDefined(c, {src_arg, index_arg});
  TORCH_CHECK(src.device() == index.device(),
              "segment_sum_coo: src and index must be on the same device "
              "(got src=",
              src.device(), ", index=", index.device(), ")");
  if (out.has_value()) {
    at::TensorArg out_arg{out.value(), "out", 2};
    at::checkAllDefined(c, {out_arg});
    TORCH_CHECK(src.device() == out.value().device(),
                "segment_sum_coo: src and out must be on the same device "
                "(got src=",
                src.device(), ", out=", out.value().device(), ")");
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::segment_sum_coo", "")
                       .typed<decltype(segment_sum_coo)>();
  return op.call(src, index, out, dim_size);
}

PYG_API at::Tensor segment_mean_coo(const at::Tensor& src,
                                    const at::Tensor& index,
                                    const std::optional<at::Tensor>& out,
                                    std::optional<int64_t> dim_size) {
  at::TensorArg src_arg{src, "src", 0};
  at::TensorArg index_arg{index, "index", 1};
  at::CheckedFrom c{"segment_mean_coo"};

  at::checkAllDefined(c, {src_arg, index_arg});
  TORCH_CHECK(src.device() == index.device(),
              "segment_mean_coo: src and index must be on the same device "
              "(got src=",
              src.device(), ", index=", index.device(), ")");
  if (out.has_value()) {
    at::TensorArg out_arg{out.value(), "out", 2};
    at::checkAllDefined(c, {out_arg});
    TORCH_CHECK(src.device() == out.value().device(),
                "segment_mean_coo: src and out must be on the same device "
                "(got src=",
                src.device(), ", out=", out.value().device(), ")");
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::segment_mean_coo", "")
                       .typed<decltype(segment_mean_coo)>();
  return op.call(src, index, out, dim_size);
}

PYG_API std::tuple<at::Tensor, at::Tensor> segment_min_coo(
    const at::Tensor& src,
    const at::Tensor& index,
    const std::optional<at::Tensor>& out,
    std::optional<int64_t> dim_size) {
  at::TensorArg src_arg{src, "src", 0};
  at::TensorArg index_arg{index, "index", 1};
  at::CheckedFrom c{"segment_min_coo"};

  at::checkAllDefined(c, {src_arg, index_arg});
  TORCH_CHECK(src.device() == index.device(),
              "segment_min_coo: src and index must be on the same device "
              "(got src=",
              src.device(), ", index=", index.device(), ")");
  if (out.has_value()) {
    at::TensorArg out_arg{out.value(), "out", 2};
    at::checkAllDefined(c, {out_arg});
    TORCH_CHECK(src.device() == out.value().device(),
                "segment_min_coo: src and out must be on the same device "
                "(got src=",
                src.device(), ", out=", out.value().device(), ")");
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::segment_min_coo", "")
                       .typed<decltype(segment_min_coo)>();
  return op.call(src, index, out, dim_size);
}

PYG_API std::tuple<at::Tensor, at::Tensor> segment_max_coo(
    const at::Tensor& src,
    const at::Tensor& index,
    const std::optional<at::Tensor>& out,
    std::optional<int64_t> dim_size) {
  at::TensorArg src_arg{src, "src", 0};
  at::TensorArg index_arg{index, "index", 1};
  at::CheckedFrom c{"segment_max_coo"};

  at::checkAllDefined(c, {src_arg, index_arg});
  TORCH_CHECK(src.device() == index.device(),
              "segment_max_coo: src and index must be on the same device "
              "(got src=",
              src.device(), ", index=", index.device(), ")");
  if (out.has_value()) {
    at::TensorArg out_arg{out.value(), "out", 2};
    at::checkAllDefined(c, {out_arg});
    TORCH_CHECK(src.device() == out.value().device(),
                "segment_max_coo: src and out must be on the same device "
                "(got src=",
                src.device(), ", out=", out.value().device(), ")");
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::segment_max_coo", "")
                       .typed<decltype(segment_max_coo)>();
  return op.call(src, index, out, dim_size);
}

PYG_API at::Tensor gather_coo(const at::Tensor& src,
                              const at::Tensor& index,
                              const std::optional<at::Tensor>& out) {
  at::TensorArg src_arg{src, "src", 0};
  at::TensorArg index_arg{index, "index", 1};
  at::CheckedFrom c{"gather_coo"};

  at::checkAllDefined(c, {src_arg, index_arg});
  TORCH_CHECK(src.device() == index.device(),
              "gather_coo: src and index must be on the same device "
              "(got src=",
              src.device(), ", index=", index.device(), ")");
  if (out.has_value()) {
    at::TensorArg out_arg{out.value(), "out", 2};
    at::checkAllDefined(c, {out_arg});
    TORCH_CHECK(src.device() == out.value().device(),
                "gather_coo: src and out must be on the same device "
                "(got src=",
                src.device(), ", out=", out.value().device(), ")");
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::gather_coo", "")
                       .typed<decltype(gather_coo)>();
  return op.call(src, index, out);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::segment_sum_coo(Tensor src, Tensor index, "
      "Tensor? out=None, int? dim_size=None) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::segment_mean_coo(Tensor src, Tensor index, "
      "Tensor? out=None, int? dim_size=None) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::segment_min_coo(Tensor src, Tensor index, "
      "Tensor? out=None, int? dim_size=None) -> (Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::segment_max_coo(Tensor src, Tensor index, "
      "Tensor? out=None, int? dim_size=None) -> (Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::gather_coo(Tensor src, Tensor index, Tensor? out=None) -> Tensor"));
}

}  // namespace ops
}  // namespace pyg
