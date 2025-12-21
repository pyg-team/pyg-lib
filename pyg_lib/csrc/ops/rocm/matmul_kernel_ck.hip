#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

// Composable Kernel headers for grouped GEMM
#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck/utility/data_type.hpp>

#include "pyg_lib/csrc/utils/convert.h"

namespace pyg {
namespace ops {

namespace {

// Define element-wise operations for the epilogue
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

// Define data types
using ADataType       = float;
using BDataType       = float;
using AccDataType     = float;
using CShuffleDataType = float;
using DsDataType      = ck::Tuple<>;
using EDataType       = float;

// Define layouts for row-major matrices
using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::RowMajor;
using DsLayout = ck::Tuple<>;
using ELayout = ck::tensor_layout::gemm::RowMajor;

// Define element-wise operations
using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CDEElementOp = PassThrough;

// DefaultGemmKernel_FP32 using Composable Kernel
// This is the simplified FP32 GEMM kernel for AMD GPUs
// No TF32 or architecture-specific versioning needed for AMD GPUs
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
    1,                    // NumGemmKBatch
    256,                  // BlockSize
    128,                  // MPerBlock
    64,                   // NPerBlock
    8,                    // KPerBlock
    8,                    // AK1
    2,                    // BK1
    32,                   // MPerXdl
    32,                   // NPerXdl
    2,                    // MXdlPerWave
    1,                    // NXdlPerWave
    ck::Sequence<4, 64, 1>,   // ABlockTransferThreadClusterLengths_AK0_M_AK1
    ck::Sequence<1, 0, 2>,    // ABlockTransferThreadClusterArrangeOrder
    ck::Sequence<1, 0, 2>,    // ABlockTransferSrcAccessOrder
    2,                        // ABlockTransferSrcVectorDim
    8,                        // ABlockTransferSrcScalarPerVector
    8,                        // ABlockTransferDstScalarPerVector_AK1
    true,                     // ABlockLdsExtraM
    ck::Sequence<4, 64, 1>,   // BBlockTransferThreadClusterLengths_BK0_N_BK1
    ck::Sequence<0, 2, 1>,    // BBlockTransferThreadClusterArrangeOrder
    ck::Sequence<0, 2, 1>,    // BBlockTransferSrcAccessOrder
    1,                        // BBlockTransferSrcVectorDim
    2,                        // BBlockTransferSrcScalarPerVector
    2,                        // BBlockTransferDstScalarPerVector_BK1
    true,                     // BBlockLdsExtraN
    1,                        // CShuffleMXdlPerWavePerShuffle
    1,                        // CShuffleNXdlPerWavePerShuffle
    ck::Sequence<1, 32, 1, 8>,  // CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    8>;                         // CDEBlockTransferScalarPerVector_NWaveNPerXdl

void run_grouped_gemm(const at::TensorList input,
                      const at::TensorList other,
                      const at::TensorList out,
                      bool segment) {
  const int64_t num_matrices = static_cast<int64_t>(input.size());

  // Prepare host-side argument arrays
  std::vector<const void*> p_a_vec;
  std::vector<const void*> p_b_vec;
  std::vector<std::array<const void*, 0>> p_ds_vec;
  std::vector<void*> p_e_vec;
  std::vector<ck::index_t> M_vec;
  std::vector<ck::index_t> N_vec;
  std::vector<ck::index_t> K_vec;
  std::vector<ck::index_t> stride_a_vec;
  std::vector<ck::index_t> stride_b_vec;
  std::vector<std::array<ck::index_t, 0>> stride_ds_vec;
  std::vector<ck::index_t> stride_e_vec;

  p_a_vec.reserve(num_matrices);
  p_b_vec.reserve(num_matrices);
  p_ds_vec.reserve(num_matrices);
  p_e_vec.reserve(num_matrices);
  M_vec.reserve(num_matrices);
  N_vec.reserve(num_matrices);
  K_vec.reserve(num_matrices);
  stride_a_vec.reserve(num_matrices);
  stride_b_vec.reserve(num_matrices);
  stride_ds_vec.reserve(num_matrices);
  stride_e_vec.reserve(num_matrices);

  // Fill argument vectors from input tensors
  for (size_t i = 0; i < static_cast<size_t>(num_matrices); ++i) {
    const auto& A = input[i];
    const auto& B = other[i];
    auto& E = out[i];

    ck::index_t M = static_cast<ck::index_t>(A.size(0));
    ck::index_t K = static_cast<ck::index_t>(B.size(static_cast<int>(segment)));
    ck::index_t N = static_cast<ck::index_t>(E.size(1));

    // For row-major layout:
    // A: MxK with stride K
    // B: KxN with stride N
    // E: MxN with stride N
    ck::index_t stride_a = K;
    ck::index_t stride_b = N;
    ck::index_t stride_e = N;

    p_a_vec.push_back(A.data_ptr<float>());
    p_b_vec.push_back(B.data_ptr<float>());
    p_ds_vec.push_back({});
    p_e_vec.push_back(E.data_ptr<float>());

    M_vec.push_back(M);
    N_vec.push_back(N);
    K_vec.push_back(K);

    stride_a_vec.push_back(stride_a);
    stride_b_vec.push_back(stride_b);
    stride_ds_vec.push_back({});
    stride_e_vec.push_back(stride_e);
  }

  // Create device GEMM instance
  auto gemm = DeviceGroupedGemmInstance{};

  // Create invoker
  auto invoker = gemm.MakeInvoker();

  // Create argument
  auto argument = gemm.MakeArgument(
      p_a_vec,
      p_b_vec,
      p_ds_vec,
      p_e_vec,
      M_vec,
      N_vec,
      K_vec,
      stride_a_vec,
      stride_b_vec,
      stride_ds_vec,
      stride_e_vec,
      AElementOp{},
      BElementOp{},
      CDEElementOp{});

  // Check if the device operation is supported
  TORCH_CHECK(gemm.IsSupportedArgument(argument),
              "CK grouped GEMM: device operation not supported for given arguments");

  // Get HIP stream
  hipStream_t stream = at::cuda::getCurrentHIPStream();

  // Run the grouped GEMM
  // CK StreamConfig takes (stream, time_kernel, log_level, flush_icache)
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
