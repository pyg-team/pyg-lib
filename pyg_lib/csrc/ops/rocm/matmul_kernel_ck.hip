#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

// Composable Kernel headers for grouped GEMM
// Following AMD official CK documentation:
// https://rocm.docs.amd.com/projects/composable_kernel/en/latest/
#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/device_grouped_gemm.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck/utility/data_type.hpp>

#include "pyg_lib/csrc/utils/convert.h"

namespace pyg {
namespace ops {

namespace {

// Define element-wise operations for the epilogue (PassThrough = identity)
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

// Define scalar data types following CK documentation
using ADataType        = float;
using BDataType        = float;
using AccDataType      = float;
using CShuffleDataType = float;
using DsDataType       = ck::Tuple<>;  // No D tensors (bias, etc.)
using EDataType        = float;

// Define tensor layouts for row-major matrices
// CK uses RowMajor/ColumnMajor from tensor_layout::gemm namespace
using ALayout  = ck::tensor_layout::gemm::RowMajor;
using BLayout  = ck::tensor_layout::gemm::RowMajor;
using DsLayout = ck::Tuple<>;
using ELayout  = ck::tensor_layout::gemm::RowMajor;

// Define element-wise operations for A, B, and CDE (output)
using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = PassThrough;

// Number of D tensors (bias tensors) - we have none
static constexpr ck::index_t NumDTensor = 0;

// DefaultGemmKernel_FP32 using Composable Kernel
// This is the simplified FP32 GEMM kernel for AMD GPUs
// No TF32 or architecture-specific versioning needed for AMD GPUs
// Kernel parameters are tuned for general-purpose grouped GEMM workloads
using DeviceGroupedGemmInstance = ck::tensor_operation::device::DeviceGroupedGemm_Xdl<
    ALayout,
    BLayout,
    DsLayout,
    ELayout,
    ADataType,
    BDataType,
    AccDataType,
    CShuffleDataType,
    DsDataType,
    EDataType,
    AElementOp,
    BElementOp,
    CDEElementOp,
    ck::tensor_operation::device::GemmSpecialization::Default,
    1,                    // NumPrefetch (number of software pipeline stages)
    256,                  // BlockSize (threads per block)
    128,                  // MPerBlock (M tile size)
    64,                   // NPerBlock (N tile size)
    8,                    // KPerBlock (K tile size per iteration)
    8,                    // AK1 (inner K dimension for A tensor)
    2,                    // BK1 (inner K dimension for B tensor)
    32,                   // MPerXDL (M per XDL instruction)
    32,                   // NPerXDL (N per XDL instruction)
    2,                    // MXdlPerWave (M XDL tiles per wave)
    1,                    // NXdlPerWave (N XDL tiles per wave)
    ck::Sequence<4, 64, 1>,   // ABlockTransferThreadClusterLengths_AK0_M_AK1
    ck::Sequence<1, 0, 2>,    // ABlockTransferThreadClusterArrangeOrder
    ck::Sequence<1, 0, 2>,    // ABlockTransferSrcAccessOrder
    2,                        // ABlockTransferSrcVectorDim
    8,                        // ABlockTransferSrcScalarPerVector
    8,                        // ABlockTransferDstScalarPerVector_AK1
    true,                     // ABlockLdsExtraM (extra LDS for A)
    ck::Sequence<4, 64, 1>,   // BBlockTransferThreadClusterLengths_BK0_N_BK1
    ck::Sequence<0, 2, 1>,    // BBlockTransferThreadClusterArrangeOrder
    ck::Sequence<0, 2, 1>,    // BBlockTransferSrcAccessOrder
    1,                        // BBlockTransferSrcVectorDim
    2,                        // BBlockTransferSrcScalarPerVector
    2,                        // BBlockTransferDstScalarPerVector_BK1
    true,                     // BBlockLdsExtraN (extra LDS for B)
    1,                        // CShuffleMXdlPerWavePerShuffle
    1,                        // CShuffleNXdlPerWavePerShuffle
    ck::Sequence<1, 32, 1, 8>,  // CDEBlockTransferClusterLengths
    8>;                         // CDEBlockTransferScalarPerVector

void run_grouped_gemm(const at::TensorList input,
                      const at::TensorList other,
                      const at::TensorList out,
                      bool segment) {
  const std::size_t group_count = input.size();

  // Prepare argument vectors following CK's MakeArgument API
  // Reference: device_grouped_gemm.hpp - MakeArgumentPointer signature
  std::vector<const void*> p_a_vec;
  std::vector<const void*> p_b_vec;
  std::vector<std::array<const void*, NumDTensor>> p_ds_vec;
  std::vector<void*> p_e_vec;
  std::vector<ck::tensor_operation::device::GemmDesc> gemm_descs;

  p_a_vec.reserve(group_count);
  p_b_vec.reserve(group_count);
  p_ds_vec.reserve(group_count);
  p_e_vec.reserve(group_count);
  gemm_descs.reserve(group_count);

  // Fill argument vectors from input tensors
  for (std::size_t i = 0; i < group_count; ++i) {
    const auto& A = input[i];
    const auto& B = other[i];
    const auto& E = out[i];

    // Get matrix dimensions
    ck::index_t M = static_cast<ck::index_t>(A.size(0));
    ck::index_t K = static_cast<ck::index_t>(B.size(static_cast<int>(segment)));
    ck::index_t N = static_cast<ck::index_t>(E.size(1));

    // For row-major layout, stride is the number of columns
    // A: MxK with stride K (row-major: StrideA = K)
    // B: KxN with stride N (row-major: StrideB = N)
    // E: MxN with stride N (row-major: StrideC = N)
    ck::index_t stride_a = K;
    ck::index_t stride_b = N;
    ck::index_t stride_e = N;

    // Add pointers
    p_a_vec.push_back(A.data_ptr<float>());
    p_b_vec.push_back(B.data_ptr<float>());
    p_ds_vec.push_back({});  // No D tensors
    p_e_vec.push_back(E.data_ptr<float>());

    // Create GemmDesc for this GEMM problem
    // GemmDesc structure: M_, N_, K_, stride_A_, stride_B_, stride_C_, stride_Ds_
    ck::tensor_operation::device::GemmDesc gemm_desc;
    gemm_desc.M_ = M;
    gemm_desc.N_ = N;
    gemm_desc.K_ = K;
    gemm_desc.stride_A_ = stride_a;
    gemm_desc.stride_B_ = stride_b;
    gemm_desc.stride_C_ = stride_e;  // Output stride (E tensor)
    gemm_desc.stride_Ds_ = {};       // No D tensor strides

    gemm_descs.push_back(gemm_desc);
  }

  // Create device GEMM instance
  auto gemm = DeviceGroupedGemmInstance{};

  // Create invoker using CK's factory method
  auto invoker = gemm.MakeInvoker();

  // Create argument using CK's MakeArgument API
  // Reference: device_grouped_gemm_xdl.hpp - MakeArgument
  auto argument = gemm.MakeArgument(
      p_a_vec,
      p_b_vec,
      p_ds_vec,
      p_e_vec,
      gemm_descs,
      AElementOp{},
      BElementOp{},
      CDEElementOp{});

  // Validate that the device operation supports the given arguments
  TORCH_CHECK(gemm.IsSupportedArgument(argument),
              "CK grouped GEMM: device operation not supported for given arguments. "
              "Check matrix dimensions and strides are compatible with kernel configuration.");

  // Get HIP stream from PyTorch
  hipStream_t stream = at::cuda::getCurrentHIPStream();

  // Run the grouped GEMM operation
  // StreamConfig parameters: (stream, time_kernel, log_level, flush_icache)
  float avg_time = invoker.Run(argument, ck::StreamConfig{stream, false});
  (void)avg_time;  // Suppress unused variable warning
}

void grouped_matmul_out_kernel(const at::TensorList input,
                               const at::TensorList other,
                               const at::TensorList out,
                               bool segment) {
  // AMD GPU does not have TF32 or architecture versioning concerns
  // Use unified FP32 GEMM kernel (DefaultGemmKernel_FP32)
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
