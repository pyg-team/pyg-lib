#include "segment_csr.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

PYG_API at::Tensor segment_sum_csr(const at::Tensor& src,
                                   const at::Tensor& indptr,
                                   const std::optional<at::Tensor>& out) {
  at::TensorArg src_arg{src, "src", 0};
  at::TensorArg indptr_arg{indptr, "indptr", 1};
  at::CheckedFrom c{"segment_sum_csr"};

  at::checkAllDefined(c, {src_arg, indptr_arg});
  TORCH_CHECK(src.device() == indptr.device(),
              "segment_sum_csr: src and indptr must be on the same device "
              "(got src=",
              src.device(), ", indptr=", indptr.device(), ")");
  if (out.has_value()) {
    at::TensorArg out_arg{out.value(), "out", 2};
    at::checkAllDefined(c, {out_arg});
    TORCH_CHECK(src.device() == out.value().device(),
                "segment_sum_csr: src and out must be on the same device "
                "(got src=",
                src.device(), ", out=", out.value().device(), ")");
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::segment_sum_csr", "")
                       .typed<decltype(segment_sum_csr)>();
  return op.call(src, indptr, out);
}

PYG_API at::Tensor segment_mean_csr(const at::Tensor& src,
                                    const at::Tensor& indptr,
                                    const std::optional<at::Tensor>& out) {
  at::TensorArg src_arg{src, "src", 0};
  at::TensorArg indptr_arg{indptr, "indptr", 1};
  at::CheckedFrom c{"segment_mean_csr"};

  at::checkAllDefined(c, {src_arg, indptr_arg});
  TORCH_CHECK(src.device() == indptr.device(),
              "segment_mean_csr: src and indptr must be on the same device "
              "(got src=",
              src.device(), ", indptr=", indptr.device(), ")");
  if (out.has_value()) {
    at::TensorArg out_arg{out.value(), "out", 2};
    at::checkAllDefined(c, {out_arg});
    TORCH_CHECK(src.device() == out.value().device(),
                "segment_mean_csr: src and out must be on the same device "
                "(got src=",
                src.device(), ", out=", out.value().device(), ")");
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::segment_mean_csr", "")
                       .typed<decltype(segment_mean_csr)>();
  return op.call(src, indptr, out);
}

PYG_API std::tuple<at::Tensor, at::Tensor> segment_min_csr(
    const at::Tensor& src,
    const at::Tensor& indptr,
    const std::optional<at::Tensor>& out) {
  at::TensorArg src_arg{src, "src", 0};
  at::TensorArg indptr_arg{indptr, "indptr", 1};
  at::CheckedFrom c{"segment_min_csr"};

  at::checkAllDefined(c, {src_arg, indptr_arg});
  TORCH_CHECK(src.device() == indptr.device(),
              "segment_min_csr: src and indptr must be on the same device "
              "(got src=",
              src.device(), ", indptr=", indptr.device(), ")");
  if (out.has_value()) {
    at::TensorArg out_arg{out.value(), "out", 2};
    at::checkAllDefined(c, {out_arg});
    TORCH_CHECK(src.device() == out.value().device(),
                "segment_min_csr: src and out must be on the same device "
                "(got src=",
                src.device(), ", out=", out.value().device(), ")");
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::segment_min_csr", "")
                       .typed<decltype(segment_min_csr)>();
  return op.call(src, indptr, out);
}

PYG_API std::tuple<at::Tensor, at::Tensor> segment_max_csr(
    const at::Tensor& src,
    const at::Tensor& indptr,
    const std::optional<at::Tensor>& out) {
  at::TensorArg src_arg{src, "src", 0};
  at::TensorArg indptr_arg{indptr, "indptr", 1};
  at::CheckedFrom c{"segment_max_csr"};

  at::checkAllDefined(c, {src_arg, indptr_arg});
  TORCH_CHECK(src.device() == indptr.device(),
              "segment_max_csr: src and indptr must be on the same device "
              "(got src=",
              src.device(), ", indptr=", indptr.device(), ")");
  if (out.has_value()) {
    at::TensorArg out_arg{out.value(), "out", 2};
    at::checkAllDefined(c, {out_arg});
    TORCH_CHECK(src.device() == out.value().device(),
                "segment_max_csr: src and out must be on the same device "
                "(got src=",
                src.device(), ", out=", out.value().device(), ")");
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::segment_max_csr", "")
                       .typed<decltype(segment_max_csr)>();
  return op.call(src, indptr, out);
}

PYG_API at::Tensor gather_csr(const at::Tensor& src,
                              const at::Tensor& indptr,
                              const std::optional<at::Tensor>& out) {
  at::TensorArg src_arg{src, "src", 0};
  at::TensorArg indptr_arg{indptr, "indptr", 1};
  at::CheckedFrom c{"gather_csr"};

  at::checkAllDefined(c, {src_arg, indptr_arg});
  TORCH_CHECK(src.device() == indptr.device(),
              "gather_csr: src and indptr must be on the same device "
              "(got src=",
              src.device(), ", indptr=", indptr.device(), ")");
  if (out.has_value()) {
    at::TensorArg out_arg{out.value(), "out", 2};
    at::checkAllDefined(c, {out_arg});
    TORCH_CHECK(src.device() == out.value().device(),
                "gather_csr: src and out must be on the same device "
                "(got src=",
                src.device(), ", out=", out.value().device(), ")");
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::gather_csr", "")
                       .typed<decltype(gather_csr)>();
  return op.call(src, indptr, out);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(
      TORCH_SELECTIVE_SCHEMA("pyg::segment_sum_csr(Tensor src, Tensor indptr, "
                             "Tensor? out=None) -> Tensor"));
  m.def(
      TORCH_SELECTIVE_SCHEMA("pyg::segment_mean_csr(Tensor src, Tensor indptr, "
                             "Tensor? out=None) -> Tensor"));
  m.def(
      TORCH_SELECTIVE_SCHEMA("pyg::segment_min_csr(Tensor src, Tensor indptr, "
                             "Tensor? out=None) -> (Tensor, Tensor)"));
  m.def(
      TORCH_SELECTIVE_SCHEMA("pyg::segment_max_csr(Tensor src, Tensor indptr, "
                             "Tensor? out=None) -> (Tensor, Tensor)"));
  m.def(
      TORCH_SELECTIVE_SCHEMA("pyg::gather_csr(Tensor src, Tensor indptr, "
                             "Tensor? out=None) -> Tensor"));
}

}  // namespace ops
}  // namespace pyg
