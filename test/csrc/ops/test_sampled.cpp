#include <ATen/ATen.h>
#include <c10/core/DeviceType.h>
#include <gtest/gtest.h>

#include "pyg_lib/csrc/ops/sampled.h"

class MultipleDeviceTest : public testing::TestWithParam<c10::DeviceType> {};

TEST_P(MultipleDeviceTest, SampledOpTest) {
  const auto param = ::testing::TestWithParam<c10::DeviceType>::GetParam();
  auto options = at::TensorOptions().device(param);

  at::Tensor a, b, out, grad_a, grad_b, grad_out;
  auto a_index = at::tensor({0, 1, 3}, options.dtype(at::kLong));
  auto b_index = at::tensor({3, 4, 5}, options.dtype(at::kLong));

  a = at::randn({3, 8}, options).requires_grad_();
  b = at::randn({3, 8}, options).requires_grad_();
  grad_out = at::randn({3, 8}, options);
  out = pyg::ops::sampled_op(a, b, c10::nullopt, c10::nullopt, "add");
  out.backward(grad_out);
  grad_a = a.grad().clone(), grad_b = b.grad().clone();
  EXPECT_TRUE(at::allclose(out, a + b));
  a.grad().fill_(0);
  b.grad().fill_(0);
  (a + b).backward(grad_out);
  EXPECT_TRUE(at::allclose(grad_a, a.grad()));
  EXPECT_TRUE(at::allclose(grad_b, b.grad()));

  a = at::randn({6, 8}, options).requires_grad_();
  b = at::randn({3, 8}, options).requires_grad_();
  grad_out = at::randn({3, 8}, options);
  out = pyg::ops::sampled_op(a, b, a_index, c10::nullopt, "sub");
  out.backward(grad_out);
  grad_a = a.grad().clone(), grad_b = b.grad().clone();
  EXPECT_TRUE(at::allclose(out, a.index_select(0, a_index) - b));
  a.grad().fill_(0);
  b.grad().fill_(0);
  (a.index_select(0, a_index) - b).backward(grad_out);
  EXPECT_TRUE(at::allclose(grad_a, a.grad()));
  EXPECT_TRUE(at::allclose(grad_b, b.grad()));

  a = at::randn({3, 8}, options).requires_grad_();
  b = at::randn({6, 8}, options).requires_grad_();
  grad_out = at::randn({3, 8}, options);
  out = pyg::ops::sampled_op(a, b, c10::nullopt, b_index, "mul");
  out.backward(grad_out);
  grad_a = a.grad().clone(), grad_b = b.grad().clone();
  EXPECT_TRUE(at::allclose(out, a * b.index_select(0, b_index)));
  a.grad().fill_(0);
  b.grad().fill_(0);
  (a * b.index_select(0, b_index)).backward(grad_out);
  EXPECT_TRUE(at::allclose(grad_a, a.grad()));
  EXPECT_TRUE(at::allclose(grad_b, b.grad()));

  a = at::randn({6, 8}, options).requires_grad_();
  b = at::randn({8, 8}, options).requires_grad_();
  grad_out = at::randn({3, 8}, options);
  out = pyg::ops::sampled_op(a, b, a_index, b_index, "div");
  out.backward(grad_out);
  grad_a = a.grad().clone(), grad_b = b.grad().clone();
  EXPECT_TRUE(at::allclose(
      out, a.index_select(0, a_index) / b.index_select(0, b_index)));
  a.grad().fill_(0);
  b.grad().fill_(0);
  (a.index_select(0, a_index) / b.index_select(0, b_index)).backward(grad_out);
  EXPECT_TRUE(at::allclose(grad_a, a.grad()));
  EXPECT_TRUE(at::allclose(grad_b, b.grad()));
}

INSTANTIATE_TEST_SUITE_P(OpsTest,
                         MultipleDeviceTest,
#ifdef WITH_CUDA
                         testing::Values(at::kCPU, at::kCUDA));
#else
                         testing::Values(at::kCPU));
#endif
