#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/kernel/gemm_grouped.h>

namespace pyg {
namespace segment {

namespace {

at::Tensor matmul_kernel(const at::Tensor& input,
                         const at::Tensor& ptr,
                         const at::Tensor& other,
                         const at::Tensor& out) {
  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassGemm =
      cutlass::gemm::device::Gemm<float,         // Data-type of A matrix
                                  ColumnMajor,   // Layout of A matrix
                                  float,         // Data-type of B matrix
                                  ColumnMajor,   // Layout of B matrix
                                  float,         // Data-type of C matrix
                                  ColumnMajor>;  // Layout of C

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_matmul"), TORCH_FN(matmul_kernel));
}

}  // namespace segment
}  // namespace pyg
