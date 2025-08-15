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

  // 仅支持 float32（与原实现一致）；如需支持半精度/FP8，可改 typeA/B/C/D 与 compute type
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

  // op 设定：A 恒不转置；B 是否转置由 segment 决定（与原逻辑一致）
  hipblasOperation_t opA = HIPBLAS_OP_N;
  hipblasOperation_t opB = segment ? HIPBLAS_OP_T : HIPBLAS_OP_N;

  // hipBLASLt 句柄
  hipblasLtHandle_t handle = nullptr;
  HIPBLASLT_CHECK(hipblasLtCreate(&handle));

  // 创建 GroupedGemm 实例（指定 opA/opB 与数据/计算类型）
  hipblaslt_ext::GroupedGemm grouped(
      handle,
      opA, opB,
      HIP_R_32F, HIP_R_32F, HIP_R_32F, HIP_R_32F,
      HIPBLAS_COMPUTE_32F); // V2 构造与 setProblem 描述见官方文档。:contentReference[oaicite:1]{index=1}

  // 为每个 GEMM 准备 (m,n,k,batch=1)、epilogue、inputs（V2）
  std::vector<int64_t> m_vec(num_mats), n_vec(num_mats), k_vec(num_mats), batch_vec(num_mats, 1);

  std::vector<hipblaslt_ext::GemmEpilogueV2> epi_vec(num_mats); // 默认为 DEFAULT
  std::vector<hipblaslt_ext::GemmInputsV2>   in_vec(num_mats);

  // alpha/beta 存在 host 上，传指针（生命周期覆盖本函数）
  std::vector<float> alpha(num_mats, 1.0f), beta(num_mats, 0.0f);

  for (int64_t i = 0; i < num_mats; ++i) {
    const auto A = input[i];
    const auto B = other[i];
    const auto D = out[i];

    const int64_t m = A.size(0);
    const int64_t n = D.size(1);
    // 与原代码保持一致：k = other.size((int)segment)
    const int64_t k = B.size(static_cast<int>(segment));

    // 形状基本检查
    TORCH_CHECK(A.size(1) == k, "A(m,k) mismatch: A.size(1)=", A.size(1), " vs k=", k);
    if (segment) {
      // B 视为给的是 (n,k)，通过 opB=T 作为 (k,n) 使用
      TORCH_CHECK(B.size(0) == n && B.size(1) == k, "B(n,k) expected when segment=true");
    } else {
      // B 视为 (k,n)，opB=N
      TORCH_CHECK(B.size(0) == k && B.size(1) == n, "B(k,n) expected when segment=false");
    }
    TORCH_CHECK(D.size(0) == m, "D(m,n) mismatch on dim0");

    m_vec[i] = m;
    n_vec[i] = n;
    k_vec[i] = k;

    // V2 Inputs
    in_vec[i].setA(A.data_ptr<float>());
    in_vec[i].setB(B.data_ptr<float>());
    in_vec[i].setC(D.data_ptr<float>()); // 与原实现一致：C 与 D 同址（beta=0 时等价）
    in_vec[i].setD(D.data_ptr<float>());
    in_vec[i].setAlpha(&alpha[i]);
    in_vec[i].setBeta(&beta[i]);
    // 如需 bias/aux/scale/amaxD，可在此补充 V2 的 setter
  }

  // 设定问题（V2 的批量 API）
  HIPBLASLT_CHECK(grouped.setProblem(m_vec, n_vec, k_vec, batch_vec, epi_vec, in_vec));  // :contentReference[oaicite:2]{index=2}

  // 要求/限制工作区大小（V2 Preference）
  hipblaslt_ext::GemmPreferenceV2 pref;
  pref.setMaxWorkspaceBytes(1 << 20); // 1MB，按官方示例与经验值，可按需调大。:contentReference[oaicite:3]{index=3}

  // 基于实例获取启发式算法
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
  HIPBLASLT_CHECK(grouped.algoGetHeuristic(/*requestedAlgoCount=*/64, pref, heuristic)); // :contentReference[oaicite:4]{index=4}
  TORCH_CHECK(!heuristic.empty(), "No heuristic algorithm found for grouped GEMM.");

  // 选择可用的第一个算法，并获取所需 workspace 大小
  size_t workspace_bytes = 0;
  int picked = -1;
  for (int j = 0; j < (int)heuristic.size(); ++j) {
    if (grouped.isAlgoSupported(heuristic[j].algo, workspace_bytes) == HIPBLAS_STATUS_SUCCESS) {
      picked = j;
      break;
    }
  }
  TORCH_CHECK(picked >= 0, "No supported algorithm found for grouped GEMM.");

  // 分配 workspace（需 16B 对齐；PyTorch 分配通常已满足对齐需求）
  at::Tensor workspace;
  if (workspace_bytes > 0) {
    workspace = at::empty({(long long)workspace_bytes}, out[0].options().dtype(at::kByte));
  }

  // 使用当前 HIP stream
  hipStream_t stream = at::cuda::getCurrentHIPStream();

  // 初始化并运行
  HIPBLASLT_CHECK(grouped.initialize(heuristic[picked].algo,
                                     workspace_bytes ? workspace.data_ptr() : nullptr,
                                     /*useUserArgs=*/true,
                                     stream)); // :contentReference[oaicite:5]{index=5}
  HIPBLASLT_CHECK(grouped.run(stream)); // 普通路径（非 UserArguments 外置）:contentReference[oaicite:6]{index=6}

  // 清理
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

TORCH_LIBRARY_IMPL(pyg, HIP, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::grouped_matmul"),
         TORCH_FN(grouped_matmul_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_matmul"),
         TORCH_FN(segment_matmul_kernel));
}

}  // namespace ops
}  // namespace pyg
