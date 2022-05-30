#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

#include <cutlass/gemm/gemm.h>

namespace pyg {
namespace segment {

namespace {

at::Tensor matmul_kernel(const at::Tensor& input,
                         const at::Tensor& ptr,
                         const at::Tensor& other,
                         const at::Tensor& out) {
  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_matmul"), TORCH_FN(matmul_kernel));
}

}  // namespace segment
}  // namespace pyg
