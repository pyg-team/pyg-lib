#include <torch/library.h>

TORCH_LIBRARY(pyg, m) {
  m.def(
      "spmm(Tensor rowptr, Tensor col, Tensor? optional_value, Tensor mat, str "
      "reduce) -> (Tensor, Tensor?)");
  m.def(
      "spmm_value_bw(Tensor row, Tensor rowptr, Tensor col, Tensor mat, Tensor "
      "grad, str reduce) -> Tensor");
}
