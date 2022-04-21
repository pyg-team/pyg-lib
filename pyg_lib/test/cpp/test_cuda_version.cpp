#include <gtest/gtest.h>

#include <pyg/library.h>

TEST(CudaVersionTest, BasicAssertions) {
  // CPU detect no cuda
  EXPECT_EQ(pyg::cuda_version(), (int64_t)-1);
}
