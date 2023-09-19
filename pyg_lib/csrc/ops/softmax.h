#pragma once

#include <ATen/ATen.h>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace ops {

// Performs softmax operations for each group.
PYG_API at::Tensor softmax_forward(const at::Tensor& src,
                                   const at::optional<at::Tensor> index,
                                   const at::optional<at::Tensor> ptr,
                                   const at::optional<int64_t> num_nodes,
                                   const int64_t dim = 0);

// Computes gradient for grouped softmax operations.
PYG_API at::Tensor softmax_backward(const at::Tensor& out,
                                    const at::Tensor& out_grad,
                                    const at::optional<at::Tensor> index,
                                    const at::optional<at::Tensor> ptr,
                                    const at::optional<int64_t> num_nodes,
                                    const int64_t dim = 0);

}  // namespace ops
}  // namespace pyg
