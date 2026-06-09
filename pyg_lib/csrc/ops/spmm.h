#pragma once

#include <ATen/ATen.h>
#include <optional>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace ops {

// Computes a CSR sparse-dense matrix multiplication with sum reduction:
//
//   out[..., row, :] = sum_{e in rowptr[row]:rowptr[row + 1]}
//       value[e] * mat[..., col[e], :]
//
// When `value` is not provided, all sparse entries are treated as one.
// `rowptr` and `col` are 1-D int64 tensors. `mat` is at least 2-D with the
// sparse column dimension at `mat.size(-2)` and feature dimension at
// `mat.size(-1)`. Leading dimensions of `mat` are treated as batches.
PYG_API at::Tensor spmm_sum(const at::Tensor& rowptr,
                            const at::Tensor& col,
                            const std::optional<at::Tensor>& value,
                            const at::Tensor& mat);

}  // namespace ops
}  // namespace pyg
