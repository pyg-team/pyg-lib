#include <gtest/gtest.h>

#include <pyg/library.h>

TEST(CudaVersionTest, BasicAssertions) {
#ifdef TEST_WITH_CUDA
  EXPECT_NE(pyg::cuda_version(), (int64_t)-1);
#else
  EXPECT_EQ(pyg::cuda_version(), (int64_t)-1);
#endif
}
