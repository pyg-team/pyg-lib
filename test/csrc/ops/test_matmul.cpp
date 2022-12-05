#include <ATen/ATen.h>
#include <c10/core/DeviceType.h>
#include <gtest/gtest.h>

#include "pyg_lib/csrc/ops/matmul.h"

class MultipleDeviceTest : public testing::TestWithParam<c10::DeviceType> {};

TEST_P(MultipleDeviceTest, GroupedMatmulForward) {
  const auto param = ::testing::TestWithParam<c10::DeviceType>::GetParam();
  auto options = at::TensorOptions().device(param);

  std::vector<at::Tensor> input{at::randn({5, 8}, options),
                                at::randn({3, 12}, options)};
  std::vector<at::Tensor> other{at::randn({8, 16}, options),
                                at::randn({12, 32}, options)};

  auto out = pyg::ops::grouped_matmul(input, other);
  EXPECT_EQ(out[0].size(0), 5);
  EXPECT_EQ(out[0].size(1), 16);
  EXPECT_EQ(out[1].size(0), 3);
  EXPECT_EQ(out[1].size(1), 32);
  auto expected_out0 = at::matmul(input[0], other[0]);
  EXPECT_TRUE(at::allclose(out[0], expected_out0, 0.1, 0.1));
  auto expected_out1 = at::matmul(input[1], other[1]);
  EXPECT_TRUE(at::allclose(out[1], expected_out1, 0.1, 0.1));
}

TEST_P(MultipleDeviceTest, SegmentMatmulForward) {
  const auto param = ::testing::TestWithParam<c10::DeviceType>::GetParam();
  auto options = at::TensorOptions().device(param);

  auto input = at::randn({8, 12}, options);
  auto ptr = at::tensor({0, 5, 8}, options.dtype(at::kLong));
  auto other = at::randn({2, 12, 16}, options);

  auto out = pyg::ops::segment_matmul(input, ptr, other);
  EXPECT_EQ(out.size(0), 8);
  EXPECT_EQ(out.size(1), 16);
  auto expected_out0 = at::matmul(input.narrow(0, 0, 5), other[0]);
  EXPECT_TRUE(at::allclose(out.narrow(0, 0, 5), expected_out0, 0.1, 0.1));
  auto expected_out1 = at::matmul(input.narrow(0, 5, 3), other[1]);
  EXPECT_TRUE(at::allclose(out.narrow(0, 5, 3), expected_out1, 0.1, 0.1));
}

TEST_P(MultipleDeviceTest, SegmentMatmulBackward) {
  const auto param = ::testing::TestWithParam<c10::DeviceType>::GetParam();
  auto options = at::TensorOptions().device(param);

  auto input = at::randn({8, 12}, options).requires_grad_();
  auto ptr = at::tensor({0, 5, 8}, options.dtype(at::kLong));
  auto other = at::randn({2, 12, 16}, options).requires_grad_();

  auto out = pyg::ops::segment_matmul(input, ptr, other);
  out.mean().backward();
  EXPECT_TRUE(input.grad().numel() == input.numel());
  EXPECT_TRUE(other.grad().numel() == other.numel());
}

INSTANTIATE_TEST_SUITE_P(OpsTest,
                         MultipleDeviceTest,
#ifdef WITH_CUDA
                         testing::Values(at::kCPU, at::kCUDA));
#else
                         testing::Values(at::kCPU));
#endif
