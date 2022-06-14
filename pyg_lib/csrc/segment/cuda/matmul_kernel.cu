#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/kernel/gemm_grouped.h>
#include <cutlass/util/command_line.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/util/distribution.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/gemm_complex.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/reference/host/gemm_complex.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_copy.h>
#include <cutlass/util/reference/host/tensor_norm.h>
#include <cutlass/util/tensor_view_io.h>

namespace pyg {
namespace segment {

namespace {

void grouped_matmul_out_kernel(const std::vector<at::Tensor>& input,
                               const std::vector<at::Tensor>& other,
                               const std::vector<at::Tensor>& out) {}

std::vector<at::Tensor> grouped_matmul_kernel(
    const std::vector<at::Tensor>& input,
    const std::vector<at::Tensor>& other) {
  // TODO (matthias) Check tensor devices.
  // TODO (matthias) Check for contiguous memory.

  std::vector<at::Tensor> out(input.size());
  for (size_t i = 0; i < input.size(); ++i)
    out[i] = input[i].new_empty({input[i].size(0), other[i].size(-1)});

  grouped_matmul_out_kernel(input, other, out);

  return out;
}

at::Tensor segment_matmul_kernel(const at::Tensor& input,
                                 const at::Tensor& ptr,
                                 const at::Tensor& other) {
  // TODO (matthias) Check tensor devices.
  // TODO (matthias) Check for contiguous memory.

  auto size = ptr.narrow(/*dim=*/0, /*start=*/1, /*length=*/ptr.numel() - 1) -
              ptr.narrow(/*dim=*/0, /*start=*/0, /*length=*/ptr.numel() - 1);
  size = size.cpu();  // `at::split` requires CPU-allocated array.
  // TODO (matthias) Allow other types than `int64_t`.
  auto sizes = at::IntArrayRef(size.data_ptr<int64_t>(), size.numel());

  const auto out = input.new_empty({input.size(0), other.size(-1)});

  grouped_matmul_out_kernel(
      input.split_with_sizes(/*split_size=*/sizes, /*dim=*/0),
      other.split(/*split_size=*/1, /*dim=*/0),
      out.split_with_sizes(/*split_size=*/sizes, /*dim=*/0));

  return out;
}

// // TODO: Requires contiguous memory!
// auto num_matrices = ptr.numel() - 1;

// using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
//     float,                                         //
//     cutlass::layout::RowMajor,                     //
//     cutlass::ComplexTransform::kNone,              //
//     8,                                             //
//     float,                                         //
//     cutlass::layout::RowMajor,                     //
//     cutlass::ComplexTransform::kNone,              //
//     8,                                             //
//     float,                                         //
//     cutlass::layout::RowMajor,                     //
//     float,                                         //
//     cutlass::arch::OpClassTensorOp,                //
//     cutlass::arch::Sm80,                           //
//     cutlass::gemm::GemmShape<256, 128, 32>,        //
//     cutlass::gemm::GemmShape<64, 64, 32>,          //
//     cutlass::gemm::GemmShape<16, 8, 8>,            //
//     cutlass::epilogue::thread::LinearCombination<  //
//         float,
//         8,
//         float,
//         float>,                                                     //
//     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,  //
//     2,                                                              //
//     cutlass::arch::OpMultiplyAdd                                    //
//     >::GemmKernel;

// // auto ptr_data = ptr.cpu().data_ptr<int64_t>();

// std::vector<float*> ptr_A_host(num_matrices);
// std::vector<float*> ptr_B_host(num_matrices);
// std::vector<float*> ptr_D_host(num_matrices);

// for (size_t i = 0; i < num_matrices; ++i) {
//   ptr_A_host[i] = input[i].data_ptr<float>();
//   ptr_B_host[i] = other[i].data_ptr<float>();
//   ptr_D_host[i] = out[i].data_ptr<float>();
//   // ptr_A_host[i] = input.data_ptr<float>() + (ptr_data[i] *
//   // input.size(1)); ptr_D_host[i] = out.data_ptr<float>() + (ptr_data[i] *
//   // out.size(1));
// }

// cutlass::DeviceAllocation<float*> ptr_A;
// ptr_A.reset(num_matrices);
// ptr_A.copy_from_host(ptr_A_host.data());

// cutlass::DeviceAllocation<float*> ptr_B;
// ptr_B.reset(num_matrices);
// ptr_B.copy_from_host(ptr_B_host.data());

// cutlass::DeviceAllocation<float*> ptr_D;
// ptr_D.reset(num_matrices);
// ptr_D.copy_from_host(ptr_D_host.data());

// std::vector<cutlass::gemm::GemmCoord> all_problems(num_matrices);
// std::vector<int64_t> lda_host(num_matrices);
// std::vector<int64_t> ldb_host(num_matrices);
// std::vector<int64_t> ldd_host(num_matrices);
// for (size_t i = 0; i < num_matrices; ++i) {
//   auto m = input.size(1);
//   auto k = input.size(2);
//   auto n = out.size(2);
//   all_problems[i] = cutlass::gemm::GemmCoord(m, n, k);
//   lda_host[i] = GemmKernel::LayoutA::packed({m, k}).stride(0);
//   ldb_host[i] = GemmKernel::LayoutB::packed({k, n}).stride(0);
//   ldd_host[i] = GemmKernel::LayoutC::packed({m, n}).stride(0);
// }

// cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> all_problems_device;
// all_problems_device.reset(num_matrices);
// all_problems_device.copy_from_host(all_problems.data());

// cutlass::DeviceAllocation<int64_t> lda;
// lda.reset(num_matrices);
// lda.copy_from_host(lda_host.data());

// cutlass::DeviceAllocation<int64_t> ldb;
// ldb.reset(num_matrices);
// ldb.copy_from_host(ldb_host.data());

// cutlass::DeviceAllocation<int64_t> ldd;
// ldd.reset(num_matrices);
// ldd.copy_from_host(ldd_host.data());

// /* configurate the GEMM args */
// using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
// typename EpilogueOutputOp::Params epilogue_op(1.0, 0.0);

// using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
// int threadblock_count = 1024;
// typename GemmGrouped::Arguments args(all_problems_device.get(),
//                                      num_matrices,
//                                      threadblock_count,
//                                      epilogue_op,
//                                      ptr_A.get(),
//                                      ptr_B.get(),
//                                      ptr_D.get(),
//                                      ptr_D.get(),
//                                      lda.get(),
//                                      ldb.get(),
//                                      ldd.get(),
//                                      ldd.get());

// GemmGrouped gemm;
// cutlass::Status status;
// status = gemm.initialize(args);
// TORCH_CHECK(status == cutlass::Status::kSuccess,
//             "GroupedGEMM kernel initialization: failed \n");
// status = gemm.run();
// TORCH_CHECK(status == cutlass::Status::kSuccess,
//             "GroupedGEMM kernel run: failed \n");

// return out;
// }

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::grouped_matmul"),
         TORCH_FN(grouped_matmul_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_matmul"),
         TORCH_FN(segment_matmul_kernel));
}

}  // namespace segment
}  // namespace pyg
