#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "pyg_lib/csrc/ops/matmul.h"

#ifdef WITH_CUDA
TEST(GroupedMatmulTest, BasicAssertions) {
  // TODO (matthias) skip for now due to missing dispatcher support.
  return;
  auto options = at::TensorOptions().device(at::kCUDA);

  std::vector<at::Tensor> input{at::randn({5, 8}, options),
                                at::randn({3, 12}, options)};
  std::vector<at::Tensor> other{at::randn({8, 16}, options),
                                at::randn({12, 32}, options)};

  auto out = pyg::ops::grouped_matmul(input, other);
  EXPECT_EQ(out[0].size(0), 5);
  EXPECT_EQ(out[0].size(1), 16);
  EXPECT_EQ(out[1].size(0), 3);
  EXPECT_EQ(out[1].size(1), 32);
  EXPECT_TRUE(at::allclose(out[0], at::matmul(input[0], other[0]), 1e-01));
  EXPECT_TRUE(at::allclose(out[1], at::matmul(input[1], other[1]), 1e-01));
}
#endif

#ifdef WITH_CUDA
TEST(SegmentMatmulTest, BasicAssertions) {
  auto options = at::TensorOptions().device(at::kCUDA);

  auto input = at::randn({8, 12}, options);
  auto ptr = at::tensor({0, 5, 8}, options.dtype(at::kLong));
  auto other = at::randn({2, 12, 16}, options);

  auto out = pyg::ops::segment_matmul(input, ptr, other);
  EXPECT_EQ(out.size(0), 8);
  EXPECT_EQ(out.size(1), 16);
  EXPECT_TRUE(at::allclose(out.narrow(0, 0, 5),
                           at::matmul(input.narrow(0, 0, 5), other[0]), 1e-01));
  EXPECT_TRUE(at::allclose(out.narrow(0, 5, 3),
                           at::matmul(input.narrow(0, 5, 3), other[1]), 1e-01));
}
#endif
