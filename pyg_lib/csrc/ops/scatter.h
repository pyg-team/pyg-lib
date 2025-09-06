#pragma once

#include <ATen/ATen.h>

#include "../macros.h"

namespace pyg {
namespace ops {

// Scatter add operation with optimized CUDA kernels
PYG_API at::Tensor scatter_add(const at::Tensor& src,
                               const at::Tensor& index,
                               const int64_t dim,
                               const at::optional<at::Tensor> out,
                               const at::optional<int64_t> dim_size);

// Scatter mean operation with optimized CUDA kernels  
PYG_API at::Tensor scatter_mean(const at::Tensor& src,
                                const at::Tensor& index,
                                const int64_t dim,
                                const at::optional<at::Tensor> out,
                                const at::optional<int64_t> dim_size);

}  // namespace ops
}  // namespace pyg