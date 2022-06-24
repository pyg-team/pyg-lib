#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/util/host_tensor.h>

namespace pyg {
namespace ops {

namespace {

void grouped_matmul_out_kernel(const std::vector<at::Tensor>& input,
                               const std::vector<at::Tensor>& other,
                               const std::vector<at::Tensor>& out) {
  // TODO (matthias) Check tensor devices.
  // TODO (matthias) Check for contiguous memory.

  // TODO (matthias) Allow for other types than `float`.
  // TODO (matthias) Are these attributes correctly set?
  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      float,                                         // Element A
      cutlass::layout::RowMajor,                     // Layout A
      cutlass::ComplexTransform::kNone,              //
      8,                                             // Granularity A
      float,                                         // Element B
      cutlass::layout::RowMajor,                     // Layout B
      cutlass::ComplexTransform::kNone,              //
      8,                                             // Granularity B
      float,                                         // Element C&D
      cutlass::layout::RowMajor,                     // Layout C&D
      float,                                         // Element Accumulator
      cutlass::arch::OpClassTensorOp,                // Operator Class Tag
      cutlass::arch::Sm80,                           // Architecture
      cutlass::gemm::GemmShape<256, 128, 32>,        // Threadblock-level Tile
      cutlass::gemm::GemmShape<64, 64, 32>,          // Warp-level Tile
      cutlass::gemm::GemmShape<16, 8, 8>,            // Warp-level Tile
      cutlass::epilogue::thread::LinearCombination<  // Epilogue
          float, 8, float, float>,                   //
      cutlass::gemm::threadblock::                   // Swizzling Operator
      GemmIdentityThreadblockSwizzle<8>,             //
      2,                                             // Stages
      cutlass::arch::OpMultiplyAdd                   // Operation
      >::GemmKernel;

  auto num_matrices = input.size();

  std::vector<float*> ptr_A_host(num_matrices);
  std::vector<float*> ptr_B_host(num_matrices);
  std::vector<float*> ptr_C_host(num_matrices);

  for (size_t i = 0; i < num_matrices; ++i) {
    ptr_A_host[i] = input[i].data_ptr<float>();
    ptr_B_host[i] = other[i].data_ptr<float>();
    ptr_C_host[i] = out[i].data_ptr<float>();
  }

  cutlass::DeviceAllocation<float*> ptr_A;
  ptr_A.reset(num_matrices);
  ptr_A.copy_from_host(ptr_A_host.data());

  cutlass::DeviceAllocation<float*> ptr_B;
  ptr_B.reset(num_matrices);
  ptr_B.copy_from_host(ptr_B_host.data());

  cutlass::DeviceAllocation<float*> ptr_C;
  ptr_C.reset(num_matrices);
  ptr_C.copy_from_host(ptr_C_host.data());

  std::vector<cutlass::gemm::GemmCoord> all_problems(num_matrices);
  std::vector<int64_t> ld_A_host(num_matrices);
  std::vector<int64_t> ld_B_host(num_matrices);
  std::vector<int64_t> ld_C_host(num_matrices);

  for (size_t i = 0; i < num_matrices; ++i) {
    auto m = input[i].size(0), k = input[i].size(1), n = out[i].size(1);
    TORCH_CHECK(input[i].size(-1) == other[i].size(-2), "Shape mismatch");
    all_problems[i] = cutlass::gemm::GemmCoord(m, n, k);
    ld_A_host[i] = GemmKernel::LayoutA::packed({m, k}).stride(0);
    ld_B_host[i] = GemmKernel::LayoutB::packed({k, n}).stride(0);
    ld_C_host[i] = GemmKernel::LayoutC::packed({m, n}).stride(0);
  }

  cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> all_problems_device;
  all_problems_device.reset(num_matrices);
  all_problems_device.copy_from_host(all_problems.data());

  cutlass::DeviceAllocation<int64_t> ld_A;
  ld_A.reset(num_matrices);
  ld_A.copy_from_host(ld_A_host.data());

  cutlass::DeviceAllocation<int64_t> ld_B;
  ld_B.reset(num_matrices);
  ld_B.copy_from_host(ld_B_host.data());

  cutlass::DeviceAllocation<int64_t> ld_C;
  ld_C.reset(num_matrices);
  ld_C.copy_from_host(ld_C_host.data());

  using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
  typename EpilogueOutputOp::Params epilogue_op(1.0, 0.0);

  using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
  typename GemmGrouped::Arguments args(
      all_problems_device.get(), num_matrices, /*threadblock_count=*/1024,
      epilogue_op, ptr_A.get(), ptr_B.get(), ptr_C.get(), ptr_C.get(),
      ld_A.get(), ld_B.get(), ld_C.get(), ld_C.get());

  GemmGrouped gemm;
  auto status = gemm.initialize(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GroupedGEMM init failed");
  status = gemm.run();
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GroupedGEMM run failed");
}

std::vector<at::Tensor> grouped_matmul_kernel(
    const std::vector<at::Tensor>& input,
    const std::vector<at::Tensor>& other) {
  std::vector<at::Tensor> out(input.size());
  for (size_t i = 0; i < input.size(); ++i)
    out[i] = input[i].new_empty({input[i].size(0), other[i].size(-1)});

  grouped_matmul_out_kernel(input, other, out);

  return out;
}

at::Tensor segment_matmul_kernel(const at::Tensor& input,
                                 const at::Tensor& ptr,
                                 const at::Tensor& other) {
  auto size = ptr.narrow(/*dim=*/0, /*start=*/1, /*length=*/ptr.numel() - 1) -
              ptr.narrow(/*dim=*/0, /*start=*/0, /*length=*/ptr.numel() - 1);
  size = size.cpu();  // `at::split` requires CPU-allocated data.
  // TODO (matthias) Allow for other types than `int64_t`.
  auto sizes = at::IntArrayRef(size.data_ptr<int64_t>(), size.numel());

  const auto out = input.new_empty({input.size(0), other.size(-1)});

  grouped_matmul_out_kernel(
      input.split_with_sizes(/*split_size=*/sizes, /*dim=*/0),
      other.split(/*split_size=*/1, /*dim=*/0),
      out.split_with_sizes(/*split_size=*/sizes, /*dim=*/0));

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
