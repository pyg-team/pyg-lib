#pragma once

#include <ATen/ATen.h>
#include <torch/script.h>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace ops {

PYG_API std::tuple<at::Tensor, at::Tensor> index_sort(
    const at::Tensor& input,
    const at::optional<int64_t> max);

}  // namespace ops
}  // namespace pyg
