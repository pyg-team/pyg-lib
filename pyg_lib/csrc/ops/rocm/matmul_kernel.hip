#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <torch/library.h>
#include <torch/version.h>

#include "pyg_lib/csrc/utils/convert.h"

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
              "Grouped GEMM: list sizes mismatch");

  const int64_t G = input.size();

  hipblasLtHandle_t ltHandle;
  TORCH_CHECK(hipblasLtCreate(&ltHandle) == HIPBLAS_STATUS_SUCCESS,
              "hipBLASLt: create handle failed");

  std::vector<int64_t> m(G), n(G), k(G), batch(G, 1);
  std::vector<int64_t> lda(G), ldb(G), ldc(G), ldd(G);
  std::vector<int64_t> strideA(G, 0), strideB(G, 0), strideC(G, 0), strideD(G, 0);
  std::vector<hipblaslt_ext::GemmInputs> inputs(G);
  std::vector<hipblaslt_ext::GemmEpilogue> epis(G);

  static float alpha = 1.0f, beta = 0.0f;

  for (int64_t i = 0; i < G; ++i) {
    at::Tensor A = input[i].contiguous();
    at::Tensor B = other[i].contiguous();
    at::Tensor D = out[i].contiguous();

    int64_t mi = A.size(0);
    int64_t ki = B.size(static_cast<int>(segment));
    int64_t ni = D.size(1);

    m[i] = mi; n[i] = ni; k[i] = ki;

    lda[i] = ki;
    ldb[i] = ni;
    ldc[i] = ni;
    ldd[i] = ni;

    inputs[i].setA(A.data_ptr<float>());
    inputs[i].setB(B.data_ptr<float>());
    inputs[i].setC(D.data_ptr<float>());
    inputs[i].setD(D.data_ptr<float>());
    inputs[i].setAlpha(&alpha);
    inputs[i].setBeta(&beta);
  }

  hipblaslt_ext::Gemm gemm(
      ltHandle,
      HIPBLAS_OP_N, HIPBLAS_OP_N,
      HIP_R_32F, HIP_R_32F, HIP_R_32F, HIP_R_32F,
      HIPBLAS_COMPUTE_32F);

  hipblaslt_ext::GemmProblemType problemtype;
  problemtype.setTransposeA(HIPBLAS_OP_N);
  problemtype.setTransposeB(HIPBLAS_OP_N);
  problemtype.setAType(HIP_R_32F);
  problemtype.setBType(HIP_R_32F);
  problemtype.setCType(HIP_R_32F);
  problemtype.setDType(HIP_R_32F);
  problemtype.setComputeType(HIPBLAS_COMPUTE_32F);

  hipblasStatus_t s = gemm.setProblem(
      m, n, k, batch,
      lda, ldb, ldc, ldd,
      strideA, strideB, strideC, strideD,
      epis, inputs, problemtype);
  TORCH_CHECK(s == HIPBLAS_STATUS_SUCCESS, "hipBLASLt setProblem failed");

  hipblaslt_ext::GemmPreference pref;
  pref.setMaxWorkspaceBytes(64ull << 20);
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
  s = gemm.algoGetHeuristic(gemm, pref, heuristic);
  TORCH_CHECK(s == HIPBLAS_STATUS_SUCCESS && !heuristic.empty(),
              "hipBLASLt algoGetHeuristic failed or empty");

  size_t ws_bytes = 0;
  s = gemm.isAlgoSupported(heuristic[0].algo, ws_bytes);
  TORCH_CHECK(s == HIPBLAS_STATUS_SUCCESS, "hipBLASLt isAlgoSupported failed");

  at::Tensor ws;
  void* ws_ptr = nullptr;
  if (ws_bytes > 0) {
    ws = at::empty({static_cast<long>(ws_bytes)}, out[0].options().dtype(at::kByte));
    ws_ptr = ws.data_ptr<void>();
  }

  hipStream_t stream = at::cuda::getCurrentHIPStream().stream();
  s = gemm.initialize(heuristic[0].algo, ws_ptr, /*useUserArgs=*/true, stream);
  TORCH_CHECK(s == HIPBLAS_STATUS_SUCCESS, "hipBLASLt initialize failed");

  s = gemm.run(stream);
  TORCH_CHECK(s == HIPBLAS_STATUS_SUCCESS, "hipBLASLt run failed");

  hipblasLtDestroy(ltHandle);
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

TORCH_LIBRARY_IMPL(pyg, HIP, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::grouped_matmul"),
         TORCH_FN(grouped_matmul_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_matmul"),
         TORCH_FN(segment_matmul_kernel));
}

}  // namespace ops
}  // namespace pyg
