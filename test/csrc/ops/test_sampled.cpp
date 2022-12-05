#include <ATen/ATen.h>
#include <c10/core/DeviceType.h>
#include <gtest/gtest.h>

#include "pyg_lib/csrc/ops/sampled.h"

class MultipleDeviceTest : public testing::TestWithParam<c10::DeviceType> {};

TEST_P(MultipleDeviceTest, SampledOpTest) {
  const auto param = ::testing::TestWithParam<c10::DeviceType>::GetParam();
  auto options = at::TensorOptions().device(param);

  auto left = at::randn({4, 8}, options);
  auto right = at::randn({6, 8}, options);
  auto left_index = at::tensor({0, 1, 3}, options.dtype(at::kLong));
  auto right_index = at::tensor({3, 4, 5}, options.dtype(at::kLong));

  auto out = pyg::ops::sampled_op(left, right, left_index, right_index, "add");
  std::cout << out << std::endl;
}

INSTANTIATE_TEST_SUITE_P(OpsTest,
                         MultipleDeviceTest,
#ifdef WITH_CUDA
                         testing::Values(at::kCUDA));
#else
                         testing::Values(at::kCPU));
#endif
