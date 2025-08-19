#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <torch/library.h>
#include <torch/version.h>

#include "pyg_lib/csrc/utils/convert.h"

#define HIPBLASLT_CHECK(expr) \
  do { hipblasStatus_t _st = (expr); \
       TORCH_CHECK(_st == HIPBLAS_STATUS_SUCCESS, "hipBLASLt error: ", int(_st), " @ ", #expr); } while(0)

namespace pyg {
namespace ops {

namespace {

// Unlike CUTLASS, ROCm/hipBLASLtExt does not require and cannot instantiate a specific GemmKernel
//  type at compile time. hipBLASLt (specifically the C++ extension hipblaslt_ext) abstracts kernel selection
// into a runtime algorithm/heuristic: you give it a grouping problem (m, n, k, leading dimensions, pointers, etc.)
// and simply call algoGetHeuristic → initialize → run. The entire kernel selection and occupancy/scheduling
// process is handled within the library, without exposing the GemmKernel template type. The official
// documentation explicitly states that the extended API supports Grouped GEMMs, allowing access
// to all algorithms, heuristic selection, and execution in initialize/run. It also provides a "vectorized setProblem"
// method for describing multiple GEMMs.
void run_grouped_gemm(const at::TensorList input,
                      const at::TensorList other,
                      const at::TensorList out,
                      bool segment) {
  TORCH_CHECK(input.size() == other.size() && input.size() == out.size(),
              "input/other/out list sizes must match");
  const int64_t num_mats = input.size();
  TORCH_CHECK(num_mats > 0, "empty problem");

  for (int64_t i = 0; i < num_mats; ++i) {
    TORCH_CHECK(input[i].is_cuda() && other[i].is_cuda() && out[i].is_cuda(),
                "all tensors must be on HIP device");
    TORCH_CHECK(input[i].scalar_type() == at::kFloat &&
                other[i].scalar_type() == at::kFloat &&
                out[i].scalar_type() == at::kFloat,
                "only float32 supported in this path");
    TORCH_CHECK(input[i].is_contiguous() && other[i].is_contiguous() && out[i].is_contiguous(),
                "tensors must be contiguous for V2 simple setProblem");
  }

  hipblasOperation_t opA = HIPBLAS_OP_N;
  hipblasOperation_t opB = segment ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  hipblasLtHandle_t handle = nullptr;
  HIPBLASLT_CHECK(hipblasLtCreate(&handle));

  hipblaslt_ext::GroupedGemm grouped(
      handle,
      opA, opB,
      HIP_R_32F, HIP_R_32F, HIP_R_32F, HIP_R_32F,
      HIPBLAS_COMPUTE_32F);
  std::vector<int64_t> m_vec(num_mats), n_vec(num_mats), k_vec(num_mats), batch_vec(num_mats, 1);
  std::vector<hipblaslt_ext::GemmEpilogueV2> epi_vec(num_mats);
  std::vector<hipblaslt_ext::GemmInputsV2>   in_vec(num_mats);
  std::vector<float> alpha(num_mats, 1.0f), beta(num_mats, 0.0f);

  for (int64_t i = 0; i < num_mats; ++i) {
    const auto A = input[i];
    const auto B = other[i];
    const auto D = out[i];
    const int64_t m = A.size(0);
    const int64_t n = D.size(1);
    const int64_t k = B.size(static_cast<int>(segment));

    TORCH_CHECK(A.size(1) == k, "A(m,k) mismatch: A.size(1)=", A.size(1), " vs k=", k);
    if (segment) {
      TORCH_CHECK(B.size(0) == n && B.size(1) == k, "B(n,k) expected when segment=true");
    } else {
      TORCH_CHECK(B.size(0) == k && B.size(1) == n, "B(k,n) expected when segment=false");
    }
    TORCH_CHECK(D.size(0) == m, "D(m,n) mismatch on dim0");

    m_vec[i] = m;
    n_vec[i] = n;
    k_vec[i] = k;

    in_vec[i].setA(A.data_ptr<float>());
    in_vec[i].setB(B.data_ptr<float>());
    in_vec[i].setC(D.data_ptr<float>());
    in_vec[i].setD(D.data_ptr<float>());
    in_vec[i].setAlpha(&alpha[i]);
    in_vec[i].setBeta(&beta[i]);
  }

  HIPBLASLT_CHECK(grouped.setProblem(m_vec, n_vec, k_vec, batch_vec, epi_vec, in_vec));
  hipblaslt_ext::GemmPreferenceV2 pref;
  pref.setMaxWorkspaceBytes(1 << 20);
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
  HIPBLASLT_CHECK(grouped.algoGetHeuristic(/*requestedAlgoCount=*/64, pref, heuristic));
  TORCH_CHECK(!heuristic.empty(), "No heuristic algorithm found for grouped GEMM.");
  size_t workspace_bytes = 0;
  int picked = -1;
  for (int j = 0; j < (int)heuristic.size(); ++j) {
    if (grouped.isAlgoSupported(heuristic[j].algo, workspace_bytes) == HIPBLAS_STATUS_SUCCESS) {
      picked = j;
      break;
    }
  }
  TORCH_CHECK(picked >= 0, "No supported algorithm found for grouped GEMM.");
  at::Tensor workspace;
  if (workspace_bytes > 0) {
    workspace = at::empty({(long long)workspace_bytes}, out[0].options().dtype(at::kByte));
  }

  hipStream_t stream = at::cuda::getCurrentHIPStream();
  HIPBLASLT_CHECK(grouped.initialize(heuristic[picked].algo,
                                     workspace_bytes ? workspace.data_ptr() : nullptr,
                                     /*useUserArgs=*/true,
                                     stream));
  HIPBLASLT_CHECK(grouped.run(stream));
  HIPBLASLT_CHECK(hipblasLtDestroy(handle));
}

void grouped_matmul_out_kernel(const at::TensorList input,
                               const at::TensorList other,
                               const at::TensorList out,
                               bool segment) {
  // PyTorch doestn't support TF32 on ROCm up to now. Wait for update.
  run_grouped_gemm(input, other, out, segment);
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
  grouped_matmul_out_kernel(input_contiguous, other_contiguous, out, false);

  return out;
}

at::Tensor segment_matmul_kernel(const at::Tensor& input,
                                 const at::Tensor& ptr,
                                 const at::Tensor& other) {
  const auto size = pyg::utils::size_from_ptr(ptr).cpu();
  // TODO (matthias) Allow for other types than `int64_t`.
  const auto sizes = at::IntArrayRef(size.data_ptr<int64_t>(), size.numel());
  const auto out = input.new_empty({input.size(0), other.size(-1)});

  // TODO (matthias) Better handle non-contiguous memory layouts.
  grouped_matmul_out_kernel(
      input.contiguous().split_with_sizes(/*split_size=*/sizes, /*dim=*/0),
      other.contiguous().split(/*split_size=*/1, /*dim=*/0),
      out.split_with_sizes(/*split_size=*/sizes, /*dim=*/0), true);

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::grouped_matmul"),
         TORCH_FN(grouped_matmul_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_matmul"),
         TORCH_FN(segment_matmul_kernel));
}

}  // namespace ops
}  // namespace pyg
