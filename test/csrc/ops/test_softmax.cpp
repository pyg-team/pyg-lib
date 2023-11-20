#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include <utility>

#include "pyg_lib/csrc/ops/softmax.h"

using namespace at::indexing;

at::Tensor softmax2D_ref_impl(const at::Tensor& src,
                              const at::Tensor& ptr,
                              const int64_t dim) {
  auto out = at::zeros_like(src);

  for (int64_t i = 0; i < src.size(1 - dim); ++i) {
    for (int64_t j = 0; j < ptr.size(0) - 1; ++j) {
      const auto beg = ptr[j].item<int64_t>();
      const auto end = ptr[j + 1].item<int64_t>();
      const auto row_slice = (dim == 0) ? Slice(beg, end) : Slice(i, i + 1);
      const auto col_slice = (dim == 0) ? Slice(i, i + 1) : Slice(beg, end);
      out.index_put_({row_slice, col_slice},
                     src.index({row_slice, col_slice}).softmax(dim));
    }
  }

  return out;
}

class CPUTest : public testing::TestWithParam<int64_t> {};

TEST_P(CPUTest, SoftmaxCSRForward) {
  const auto dim = ::testing::TestWithParam<int64_t>::GetParam();
  const auto src = at::rand({8, 8});
  const auto ptr = at::tensor({0, 3, 4, 7, 8}, at::kLong);
  const auto expected_out = softmax2D_ref_impl(src, ptr, dim);

  const auto out = pyg::ops::softmax_csr(src, ptr, dim);
  EXPECT_EQ(expected_out.size(0), out.size(0));
  EXPECT_EQ(expected_out.size(1), out.size(1));
  EXPECT_TRUE(at::allclose(expected_out, out, 1e-04, 1e-04));
}

TEST_P(CPUTest, SoftmaxCSRAutogradBackward) {
  const auto dim = ::testing::TestWithParam<int64_t>::GetParam();
  const auto src = at::rand({8, 8});
  src.set_requires_grad(true);
  const auto ptr = at::tensor({0, 3, 4, 7, 8}, at::kLong);
  const auto out = softmax2D_ref_impl(src, ptr, dim);
  const auto out_grad = at::rand({8, 8});

  // use softmax_csr_backward directly
  const auto in_grad = pyg::ops::softmax_csr_backward(out, out_grad, ptr, dim);
  out.backward(out_grad);
  EXPECT_EQ(src.grad().size(0), in_grad.size(0));
  EXPECT_EQ(src.grad().size(1), in_grad.size(1));
  EXPECT_TRUE(at::allclose(src.grad(), in_grad, 1e-04, 1e-04));

  // use softmax backward via autograd module
  const auto src2 = src.detach().clone();
  src2.set_requires_grad(true);
  const auto out2 = pyg::ops::softmax_csr(src2, ptr, dim);
  out2.backward(out_grad);
  EXPECT_EQ(src.grad().size(0), src2.grad().size(0));
  EXPECT_EQ(src.grad().size(1), src2.grad().size(1));
  EXPECT_TRUE(at::allclose(src.grad(), src2.grad(), 1e-04, 1e-04));
}

INSTANTIATE_TEST_SUITE_P(OpsTest,
                         CPUTest,
                         // dim
                         testing::Values(0, 1));
