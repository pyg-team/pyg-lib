#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace ops {

// Performs softmax operations for each group.
PYG_API at::Tensor softmax_csr(const at::Tensor& src,
                               const at::Tensor& ptr,
                               const int64_t dim = 0);

// Computes gradient for grouped softmax operations.
PYG_API at::Tensor softmax_csr_backward(const at::Tensor& out,
                                        const at::Tensor& out_grad,
                                        const at::Tensor& ptr,
                                        const int64_t dim = 0);

}  // namespace ops
}  // namespace pyg
