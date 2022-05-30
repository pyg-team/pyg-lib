#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "pyg_lib/csrc/segment/matmul.h"

#ifdef WITH_CUDA
TEST(SegmentMatmulTest, BasicAssertions) {
  auto options = at::TensorOptions().device(at::kCUDA);

  auto input = at::randn({6, 8}, options);
  auto ptr = at::tensor({0, 2, 5, 6}, options.dtype(at::kLong));
  auto other = at::randn({3, 8, 16}, options);
  auto out = at::zeros({6, 8}, options);

  std::cout << input << std::endl;
  std::cout << ptr << std::endl;
  std::cout << other << std::endl;
  std::cout << out << std::endl;

  pyg::segment::matmul(input, ptr, other, out);
}
#endif
