// ROCm/HIP implementation of grouped matmul using rocBLAS
// Replaces CUTLASS-based CUDA implementation for AMD GPUs

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#include <torch/library.h>

#include "pyg_lib/csrc/utils/convert.h"

namespace pyg {
namespace ops {

namespace {

// Helper to check rocBLAS status
#define ROCBLAS_CHECK(status)                                                  \
  do {                                                                         \
    rocblas_status err = (status);                                             \
    TORCH_CHECK(err == rocblas_status_success,                                 \
                "rocBLAS error: ", rocblas_status_to_string(err));             \
  } while (0)

// Get or create rocBLAS handle for current stream
rocblas_handle get_rocblas_handle() {
  static thread_local rocblas_handle handle = nullptr;
  if (handle == nullptr) {
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
  }
  // Set stream to current HIP stream
  ROCBLAS_CHECK(rocblas_set_stream(handle, at::hip::getCurrentHIPStream()));
  return handle;
}

void grouped_matmul_out_kernel(const at::TensorList input,
                               const at::TensorList other,
                               const at::TensorList out) {
  const int64_t num_matrices = input.size();
  if (num_matrices == 0)
    return;

  rocblas_handle handle = get_rocblas_handle();
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // For small number of matrices, use individual GEMM calls
  // For larger batches, could use rocblas_gemm_batched_ex
  for (int64_t i = 0; i < num_matrices; ++i) {
    const auto& A = input[i];
    const auto& B = other[i];
    const auto& C = out[i];

    int64_t m = A.size(0);
    int64_t k = A.size(1);
    int64_t n = B.size(1);

    // rocBLAS uses column-major, but our tensors are row-major
    // C = A @ B in row-major is equivalent to C^T = B^T @ A^T in col-major
    // So we compute: C(m,n) = A(m,k) @ B(k,n)
    // In col-major: C^T(n,m) = B^T(n,k) @ A^T(k,m)
    ROCBLAS_CHECK(rocblas_sgemm(
        handle,
        rocblas_operation_none,  // B is not transposed (but we read B^T)
        rocblas_operation_none,  // A is not transposed (but we read A^T)
        n,                       // rows of op(B^T) = cols of B = n
        m,                       // cols of op(A^T) = rows of A = m
        k,                       // inner dimension
        &alpha,
        B.data_ptr<float>(),     // B in row-major = B^T in col-major
        n,                       // leading dim of B (row-major stride)
        A.data_ptr<float>(),     // A in row-major = A^T in col-major
        k,                       // leading dim of A (row-major stride)
        &beta,
        C.data_ptr<float>(),     // C in row-major = C^T in col-major
        n                        // leading dim of C (row-major stride)
    ));
  }
}

std::vector<at::Tensor> grouped_matmul_kernel(const at::TensorList input,
                                              const at::TensorList other) {
  std::vector<at::Tensor> out(input.size());
  std::vector<at::Tensor> input_contiguous(input.size());
  std::vector<at::Tensor> other_contiguous(other.size());

  for (size_t i = 0; i < input.size(); ++i) {
    input_contiguous[i] = input[i].contiguous();
    other_contiguous[i] = other[i].contiguous();
    out[i] = input[i].new_empty({input[i].size(0), other[i].size(-1)});
  }

  grouped_matmul_out_kernel(input_contiguous, other_contiguous, out);
  return out;
}

at::Tensor segment_matmul_kernel(const at::Tensor& input,
                                 const at::Tensor& ptr,
                                 const at::Tensor& other) {
  const auto size = pyg::utils::size_from_ptr(ptr).cpu();
  const auto sizes = at::IntArrayRef(size.data_ptr<int64_t>(), size.numel());
  const auto out = input.new_empty({input.size(0), other.size(-1)});

  auto input_splits = input.contiguous().split_with_sizes(sizes, 0);
  auto other_splits = other.contiguous().split(1, 0);
  auto out_splits = out.split_with_sizes(sizes, 0);

  std::vector<at::Tensor> input_vec(input_splits.begin(), input_splits.end());
  std::vector<at::Tensor> other_vec(other_splits.begin(), other_splits.end());
  std::vector<at::Tensor> out_vec(out_splits.begin(), out_splits.end());

  grouped_matmul_out_kernel(input_vec, other_vec, out_vec);
  return out;
}

}  // namespace

// Register for HIP backend (uses same "CUDA" dispatch key on ROCm)
TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::grouped_matmul"),
         TORCH_FN(grouped_matmul_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_matmul"),
         TORCH_FN(segment_matmul_kernel));
}

}  // namespace ops
}  // namespace pyg
