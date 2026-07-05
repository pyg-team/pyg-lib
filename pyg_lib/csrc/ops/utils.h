#pragma once

#include <ATen/ATen.h>

namespace pyg {
namespace ops {

// Broadcasts `src` so that its shape matches `other`. Mirrors the helper at
// `pytorch_scatter/csrc/scatter.cpp:25-33`. The result records `unsqueeze` /
// `expand` operations on the autograd graph and is used inside autograd
// `forward` methods before kernel dispatch so that the broadcast index keeps
// gradient connectivity (where applicable) and so that downstream kernels can
// assume an index tensor whose shape matches `other`.
//
// Algorithm:
//   * If `src` is 1-D, prepend `dim` singleton dims (`src.unsqueeze(0)` x dim).
//   * Append singleton dims until `src.dim() == other.dim()`.
//   * `src.expand(other.sizes())`.
//
// The returned tensor is *not* contiguous; the caller is responsible for
// `.contiguous()` if the kernel requires it.
inline at::Tensor broadcast(const at::Tensor& src,
                            const at::Tensor& other,
                            int64_t dim) {
  auto out = src;
  if (out.dim() == 1) {
    for (int64_t i = 0; i < dim; ++i)
      out = out.unsqueeze(0);
  }
  for (int64_t i = out.dim(); i < other.dim(); ++i)
    out = out.unsqueeze(-1);
  out = out.expand(other.sizes());
  return out;
}

}  // namespace ops
}  // namespace pyg
