#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <c10/util/Exception.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

// PyTorch HIP builds often define these macros globally to disable implicit
// half operators/conversions. CK's half math utilities rely on those
// conversions, so we temporarily undefine them while including CK headers.
#if defined(__HIP_NO_HALF_OPERATORS__)
#define PYG_RESTORE_HIP_NO_HALF_OPERATORS 1
#undef __HIP_NO_HALF_OPERATORS__
#endif
#if defined(__HIP_NO_HALF_CONVERSIONS__)
#define PYG_RESTORE_HIP_NO_HALF_CONVERSIONS 1
#undef __HIP_NO_HALF_CONVERSIONS__
#endif

#include <ck/ck.hpp>
#include <ck/stream_config.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>

#if defined(PYG_RESTORE_HIP_NO_HALF_OPERATORS)
#define __HIP_NO_HALF_OPERATORS__ 1
#undef PYG_RESTORE_HIP_NO_HALF_OPERATORS
#endif
#if defined(PYG_RESTORE_HIP_NO_HALF_CONVERSIONS)
#define __HIP_NO_HALF_CONVERSIONS__ 1
#undef PYG_RESTORE_HIP_NO_HALF_CONVERSIONS
#endif

#include "pyg_lib/csrc/utils/convert.h"

namespace pyg {
namespace ops {

namespace {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using DsLayout = ck::Tuple<>;
using DsDataType = ck::Tuple<>;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmDefault =
    ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmMNKPadding =
    ck::tensor_operation::device::GemmSpecialization::MNKPadding;
static constexpr ck::index_t NumDTensor = 0;

template <typename AType,
          typename BType,
          typename AccType,
          typename CType,
          ck::tensor_operation::device::GemmSpecialization GemmSpec>
using GroupedGemmKernelXdl =
    ck::tensor_operation::device::DeviceGroupedGemm_Xdl<
        Row, Col, DsLayout, Row,
        AType, BType, AccType, CType, DsDataType, CType,
        PassThrough, PassThrough, PassThrough,
        GemmSpec,
        1,   // NumGemmKPrefetchStage
        256, // BlockSize
        256, // MPerBlock
        128, // NPerBlock
        32,  // KPerBlock
        8,   // AK1
        8,   // BK1
        16,  // MPerXDL
        16,  // NPerXDL
        8,   // MXdlPerWave
        4,   // NXdlPerWave
        S<4, 64, 1>, // ABlockTransferThreadClusterLengths_AK0_M_AK1
        S<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,  // ABlockTransferSrcAccessOrder
        2,           // ABlockTransferSrcVectorDim
        8,           // ABlockTransferSrcScalarPerVector
        8,           // ABlockTransferDstScalarPerVector_AK1
        1,           // ABlockLdsAddExtraM
        S<4, 64, 1>, // BBlockTransferThreadClusterLengths_BK0_N_BK1
        S<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,  // BBlockTransferSrcAccessOrder
        2,           // BBlockTransferSrcVectorDim
        8,           // BBlockTransferSrcScalarPerVector
        8,           // BBlockTransferDstScalarPerVector_BK1
        1,           // BBlockLdsAddExtraN
        1,           // CShuffleMXdlPerWavePerShuffle
        1,           // CShuffleNXdlPerWavePerShuffle
        S<1, 32, 1, 8>, // CDEBlockTransferClusterLengths_MBlock_MPerBlock...
        4>;             // CDEBlockTransferScalarPerVector_NWaveNPerXDL

// CK official reference:
// example/15_grouped_gemm/grouped_gemm_xdl_fp16.cpp
using DefaultGroupedGemmKernel_FP16 =
    GroupedGemmKernelXdl<ck::half_t, ck::half_t, float, ck::half_t,
                         GemmDefault>;
using PaddedGroupedGemmKernel_FP16 =
    GroupedGemmKernelXdl<ck::half_t, ck::half_t, float, ck::half_t,
                         GemmMNKPadding>;

// CK official reference:
// example/15_grouped_gemm/grouped_gemm_xdl_bf16.cpp
using DefaultGroupedGemmKernel_BF16 =
    GroupedGemmKernelXdl<ck::bhalf_t, ck::bhalf_t, float, ck::bhalf_t,
                         GemmDefault>;
using PaddedGroupedGemmKernel_BF16 =
    GroupedGemmKernelXdl<ck::bhalf_t, ck::bhalf_t, float, ck::bhalf_t,
                         GemmMNKPadding>;

static_assert(sizeof(at::Half) == sizeof(ck::half_t),
              "Unexpected half type size mismatch");
static_assert(sizeof(at::BFloat16) == sizeof(ck::bhalf_t),
              "Unexpected bfloat16 type size mismatch");

bool use_ck_grouped_gemm() {
  // Default: enable CK backend.
  // Set PYG_ROCM_MATMUL_USE_CK=0 to force fallback behavior.
  const char* value = std::getenv("PYG_ROCM_MATMUL_USE_CK");
  if (value == nullptr) {
    return true;
  }
  if (std::strcmp(value, "0") == 0 || std::strcmp(value, "OFF") == 0 ||
      std::strcmp(value, "off") == 0 || std::strcmp(value, "FALSE") == 0 ||
      std::strcmp(value, "false") == 0 || std::strcmp(value, "NO") == 0 ||
      std::strcmp(value, "no") == 0) {
    return false;
  }
  return true;
}

bool require_ck_grouped_gemm() {
  // Optional strict mode:
  // Set PYG_ROCM_MATMUL_REQUIRE_CK=1 to error out instead of falling back.
  const char* value = std::getenv("PYG_ROCM_MATMUL_REQUIRE_CK");
  if (value == nullptr) {
    return false;
  }
  if (std::strcmp(value, "0") == 0 || std::strcmp(value, "OFF") == 0 ||
      std::strcmp(value, "off") == 0 || std::strcmp(value, "FALSE") == 0 ||
      std::strcmp(value, "false") == 0 || std::strcmp(value, "NO") == 0 ||
      std::strcmp(value, "no") == 0) {
    return false;
  }
  return true;
}

const char* scalar_type_name(const at::ScalarType scalar_type) {
  switch (scalar_type) {
    case at::kHalf:
      return "fp16";
    case at::kBFloat16:
      return "bf16";
    case at::kFloat:
      return "fp32";
    default:
      return "other";
  }
}

at::Tensor to_2d_other(const at::Tensor& other_tensor, bool segment) {
  if (!segment) {
    return other_tensor;
  }
  TORCH_CHECK(other_tensor.dim() == 3 && other_tensor.size(0) == 1,
              "segment_matmul expects each grouped 'other' tensor to have "
              "shape [1, K, N]");
  return other_tensor.squeeze(0);
}

void grouped_matmul_out_kernel_fallback(const at::TensorList input,
                                        const at::TensorList other,
                                        const at::TensorList out,
                                        bool segment) {
  if (input.empty()) {
    return;
  }

  const std::size_t group_count = input.size();
  for (std::size_t i = 0; i < group_count; ++i) {
    const auto& A = input[i];
    auto E = out[i];
    const auto B2d = to_2d_other(other[i], segment);
    at::mm_out(E, A, B2d);
  }
}

template <typename GemmOp, typename ATScalar>
bool run_grouped_gemm_ck_xdl(const at::TensorList input,
                             const at::TensorList other,
                             const at::TensorList out,
                             bool segment,
                             std::string& fail_reason) {
  if (input.empty()) {
    return true;
  }

  std::vector<const void*> p_a_vec;
  std::vector<const void*> p_b_vec;
  std::vector<std::array<const void*, NumDTensor>> p_ds_vec(input.size());
  std::vector<void*> p_e_vec;
  std::vector<ck::tensor_operation::device::GemmDesc> gemm_descs;
  std::vector<at::Tensor> other_ck_layout;

  p_a_vec.reserve(input.size());
  p_b_vec.reserve(input.size());
  p_e_vec.reserve(input.size());
  gemm_descs.reserve(input.size());
  other_ck_layout.reserve(input.size());

  for (std::size_t i = 0; i < input.size(); ++i) {
    const auto& A = input[i];
    const auto& E = out[i];
    const auto B2d = to_2d_other(other[i], segment);

    TORCH_CHECK(A.dim() == 2 && B2d.dim() == 2 && E.dim() == 2,
                "CK grouped GEMM expects 2D tensors");

    const ck::index_t M = static_cast<ck::index_t>(A.size(0));
    const ck::index_t K = static_cast<ck::index_t>(A.size(1));
    const ck::index_t N = static_cast<ck::index_t>(E.size(1));

    TORCH_CHECK(B2d.size(0) == K && B2d.size(1) == N,
                "Unexpected B shape for grouped GEMM");
    TORCH_CHECK(E.size(0) == M, "Unexpected output shape for grouped GEMM");

    // CK kernel is configured with BLayout=Col, so we materialize [N, K]
    // row-major (equivalent memory layout to [K, N] column-major).
    other_ck_layout.push_back(B2d.transpose(0, 1).contiguous());
    const auto& Bck = other_ck_layout.back();

    p_a_vec.push_back(A.data_ptr<ATScalar>());
    p_b_vec.push_back(Bck.data_ptr<ATScalar>());
    p_e_vec.push_back(E.data_ptr<ATScalar>());
    gemm_descs.push_back(
        {M,
         N,
         K,
         static_cast<ck::index_t>(A.stride(0)),
         static_cast<ck::index_t>(Bck.stride(0)),
         static_cast<ck::index_t>(E.stride(0)),
         {}});
  }

  try {
    auto gemm = GemmOp{};
    auto argument = gemm.MakeArgument(p_a_vec, p_b_vec, p_ds_vec, p_e_vec,
                                      gemm_descs, PassThrough{}, PassThrough{},
                                      PassThrough{});

    if (!gemm.IsSupportedArgument(argument)) {
      fail_reason = "CK kernel does not support this shape/layout";
      return false;
    }

    const auto byte_options = out[0].options().dtype(at::kByte);
    const std::size_t workspace_size = gemm.GetWorkSpaceSize(&argument);
    const std::size_t device_kargs_size = gemm.GetDeviceKernelArgSize(&argument);

    at::Tensor workspace;
    at::Tensor device_kargs;
    if (device_kargs_size > 0) {
      device_kargs =
          at::empty({static_cast<int64_t>(device_kargs_size)}, byte_options);
      gemm.SetDeviceKernelArgs(&argument, device_kargs.data_ptr());
    }
    if (workspace_size > 0 && workspace_size != device_kargs_size) {
      workspace =
          at::empty({static_cast<int64_t>(workspace_size)}, byte_options);
      gemm.SetWorkSpacePointer(&argument, workspace.data_ptr());
    }

    auto invoker = gemm.MakeInvoker();
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
    invoker.Run(argument, {stream, false});

    const hipError_t hip_err = hipGetLastError();
    if (hip_err != hipSuccess) {
      fail_reason = std::string("CK launch failed: ") + hipGetErrorString(hip_err);
      return false;
    }
    return true;
  } catch (const std::exception& e) {
    fail_reason = std::string("CK exception: ") + e.what();
    return false;
  }
}

template <typename ATScalar,
          typename DefaultGemmOp,
          typename PaddedGemmOp>
bool run_grouped_gemm_ck_xdl_best_effort(const at::TensorList input,
                                         const at::TensorList other,
                                         const at::TensorList out,
                                         bool segment,
                                         std::string& fail_reason) {
  const ck::index_t k_per_block = 32;
  const ck::index_t n_per_vector = 4;

  // Find the min/max K and whether all groups satisfy N vectorization
  // requirements for grouped CK kernels.
  ck::index_t min_k = std::numeric_limits<ck::index_t>::max();
  ck::index_t global_k = 0;
  bool all_n_aligned_for_grouped = true;
  for (std::size_t i = 0; i < input.size(); ++i) {
    const auto& a = input[i];
    const ck::index_t k = static_cast<ck::index_t>(a.size(1));
    const ck::index_t n = static_cast<ck::index_t>(out[i].size(1));
    min_k = std::min(min_k, k);
    global_k = std::max(global_k, k);
    all_n_aligned_for_grouped =
        all_n_aligned_for_grouped && (n % n_per_vector == 0);
  }

  // Only try Default / MNKPadding kernels when ALL groups have
  // K > KPerBlock, ensuring has_main_k_block_loop=true for every group.
  //
  // When any K <= KPerBlock the has_main_k_block_loop=false codepath is
  // used.  For this kernel's tile parameters (MXdlPerWave=8,
  // NXdlPerWave=4, BlockSize=256) the false path can produce silently
  // wrong results for certain M/N/K combinations, so we skip directly
  // to explicit K-padding which forces has_main_k_block_loop=true.
  std::string default_reason = "skipped (min_k <= k_per_block)";
  std::string padded_reason  = "skipped (min_k <= k_per_block)";

  if (min_k > k_per_block && all_n_aligned_for_grouped) {
    default_reason.clear();
    if (run_grouped_gemm_ck_xdl<DefaultGemmOp, ATScalar>(input, other, out,
                                                          segment,
                                                          default_reason)) {
      return true;
    }

    padded_reason.clear();
    if (run_grouped_gemm_ck_xdl<PaddedGemmOp, ATScalar>(input, other, out,
                                                         segment,
                                                         padded_reason)) {
      return true;
    }
  } else if (min_k > k_per_block) {
    default_reason = "skipped (some group has N % 4 != 0)";
    padded_reason = "skipped (some group has N % 4 != 0)";
  }

  // Runtime explicit K-padding: pad ALL groups to the same K so that every
  // group has identical has_main_k_block_loop = true.
  //
  // Background: CK's DeviceGroupedGemm_Xdl launches a single GPU kernel
  // for all groups; it requires ALL groups to share the same
  // has_main_k_block_loop value.  has_main_k_block_loop is true iff
  // ceil(K / KPerBlock) > 1, i.e. K > KPerBlock.
  //
  // The current kernel's tile parameters make the
  // has_main_k_block_loop=false codepath unreliable for certain shapes.
  // Therefore we MUST ensure the true path by padding K > KPerBlock.
  //
  // k_min_padded = 2 * KPerBlock guarantees ceil(k_padded / KPerBlock) >= 2 > 1.
  const ck::index_t k_min_padded = 2 * k_per_block;
  const ck::index_t global_k_padded =
      std::max(k_min_padded,
               ((global_k + k_per_block - 1) / k_per_block) * k_per_block);

  // For non-vectorized N, also pad N to avoid tail write corruption from
  // vectorized C output stores.
  std::string explicit_pad_reason = "not attempted";
  if (global_k_padded > 0) {
    // Run each group individually through CK instead of batching all groups
    // into a single grouped-GEMM launch. This avoids known cross-group
    // instability with heterogeneous grouped dimensions.
    bool per_group_ok = true;
    std::string per_group_fail;
    for (std::size_t i = 0; i < input.size(); ++i) {
      const auto& A = input[i];
      const auto B2d = to_2d_other(other[i], segment);

      const ck::index_t M = static_cast<ck::index_t>(A.size(0));
      const ck::index_t K = static_cast<ck::index_t>(A.size(1));
      const ck::index_t N = static_cast<ck::index_t>(B2d.size(1));
      const ck::index_t n_padded =
          ((N + n_per_vector - 1) / n_per_vector) * n_per_vector;

      auto A_pad = at::zeros({M, global_k_padded}, A.options());
      auto B_pad = at::zeros({global_k_padded, n_padded}, B2d.options());
      A_pad.narrow(1, 0, K).copy_(A);
      B_pad.narrow(0, 0, K).narrow(1, 0, N).copy_(B2d);

      const bool needs_n_pad = (n_padded != N);
      at::Tensor out_pad;
      at::Tensor out_target = out[i];
      if (needs_n_pad) {
        out_pad = at::zeros({M, n_padded}, out[i].options());
        out_target = out_pad;
      }

      std::vector<at::Tensor> sg_in = {A_pad};
      std::vector<at::Tensor> sg_oth = {segment ? B_pad.unsqueeze(0) : B_pad};
      std::vector<at::Tensor> sg_out = {out_target};

      std::string reason;
      if (!run_grouped_gemm_ck_xdl<PaddedGemmOp, ATScalar>(
              sg_in, sg_oth, sg_out, segment, reason)) {
        if (!run_grouped_gemm_ck_xdl<DefaultGemmOp, ATScalar>(
                sg_in, sg_oth, sg_out, segment, reason)) {
          per_group_ok = false;
          per_group_fail = "group " + std::to_string(i) + ": " + reason;
          break;
        }
      }

      if (needs_n_pad) {
        out[i].copy_(out_target.narrow(1, 0, N));
      }
    }
    if (per_group_ok) return true;
    explicit_pad_reason = "per-group CK failed: " + per_group_fail;
  }

  fail_reason = "default-kernel failed: " + default_reason +
                "; mnk-padding-kernel failed: " + padded_reason +
                "; explicit-k-padding failed: " + explicit_pad_reason;
  return false;
}

bool run_grouped_gemm_fp32_via_bf16(const at::TensorList input,
                                    const at::TensorList other,
                                    const at::TensorList out,
                                    bool segment,
                                    std::string& fail_reason) {
  std::vector<at::Tensor> input_bf16(input.size());
  std::vector<at::Tensor> other_bf16(other.size());
  std::vector<at::Tensor> out_bf16(out.size());

  for (std::size_t i = 0; i < input.size(); ++i) {
    input_bf16[i] = input[i].to(at::kBFloat16).contiguous();
    other_bf16[i] = other[i].to(at::kBFloat16).contiguous();
    out_bf16[i] =
        at::empty({out[i].size(0), out[i].size(1)},
                  out[i].options().dtype(at::kBFloat16));
  }

  std::string bf16_reason;
  bool ok = run_grouped_gemm_ck_xdl_best_effort<
      at::BFloat16, DefaultGroupedGemmKernel_BF16, PaddedGroupedGemmKernel_BF16>(
      input_bf16, other_bf16, out_bf16, segment, bf16_reason);
  if (!ok) {
    fail_reason = std::string("bf16-converted CK path failed: ") + bf16_reason;
    return false;
  }

  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i].copy_(out_bf16[i]);
  }
  return true;
}

bool run_grouped_gemm_fp32_via_fp16(const at::TensorList input,
                                    const at::TensorList other,
                                    const at::TensorList out,
                                    bool segment,
                                    std::string& fail_reason) {
  std::vector<at::Tensor> input_fp16(input.size());
  std::vector<at::Tensor> other_fp16(other.size());
  std::vector<at::Tensor> out_fp16(out.size());

  for (std::size_t i = 0; i < input.size(); ++i) {
    input_fp16[i] = input[i].to(at::kHalf).contiguous();
    other_fp16[i] = other[i].to(at::kHalf).contiguous();
    out_fp16[i] =
        at::empty({out[i].size(0), out[i].size(1)},
                  out[i].options().dtype(at::kHalf));
  }

  std::string fp16_reason;
  bool ok = run_grouped_gemm_ck_xdl_best_effort<
      at::Half, DefaultGroupedGemmKernel_FP16, PaddedGroupedGemmKernel_FP16>(
      input_fp16, other_fp16, out_fp16, segment, fp16_reason);
  if (!ok) {
    fail_reason = std::string("fp16-converted CK path failed: ") + fp16_reason;
    return false;
  }

  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i].copy_(out_fp16[i]);
  }
  return true;
}

void grouped_matmul_out_kernel(const at::TensorList input,
                               const at::TensorList other,
                               const at::TensorList out,
                               bool segment) {
  if (input.empty()) {
    return;
  }

  const auto scalar_type = input[0].scalar_type();
  const bool ck_enabled = use_ck_grouped_gemm();
  bool ran_ck = false;
  std::string fail_reason;

  if (ck_enabled && scalar_type == at::kHalf) {
    ran_ck = run_grouped_gemm_ck_xdl_best_effort<
        at::Half, DefaultGroupedGemmKernel_FP16, PaddedGroupedGemmKernel_FP16>(
        input, other, out, segment, fail_reason);
  } else if (ck_enabled && scalar_type == at::kBFloat16) {
    ran_ck = run_grouped_gemm_ck_xdl_best_effort<
        at::BFloat16, DefaultGroupedGemmKernel_BF16, PaddedGroupedGemmKernel_BF16>(
        input, other, out, segment, fail_reason);
  } else if (ck_enabled && scalar_type == at::kFloat) {
    std::string bf16_reason;
    ran_ck =
        run_grouped_gemm_fp32_via_bf16(input, other, out, segment, bf16_reason);
    if (!ran_ck) {
      std::string fp16_reason;
      ran_ck = run_grouped_gemm_fp32_via_fp16(input, other, out, segment,
                                              fp16_reason);
      if (!ran_ck) {
        fail_reason = bf16_reason + "; " + fp16_reason;
      }
    }
  } else if (!ck_enabled) {
    fail_reason = "CK backend disabled by PYG_ROCM_MATMUL_USE_CK";
  } else {
    fail_reason = "dtype is not currently supported by CK grouped GEMM path";
  }

  if (!ran_ck) {
    if (require_ck_grouped_gemm()) {
      TORCH_CHECK(false,
                  "pyg-lib ROCm matmul strict CK mode is enabled, but no CK "
                  "kernel accepted this input (dtype=",
                  scalar_type_name(scalar_type),
                  ", segment=",
                  segment,
                  "). Reason: ",
                  fail_reason);
    }
    TORCH_WARN(
        "pyg-lib ROCm matmul fallback: using ATen matmul because CK path was "
        "not used (dtype=",
        scalar_type_name(scalar_type),
        ", segment=",
        segment,
        "). Reason: ",
        fail_reason);
    grouped_matmul_out_kernel_fallback(input, other, out, segment);
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
