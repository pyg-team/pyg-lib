#include <gtest/gtest.h>

#include "pyg_lib/csrc/library.h"

TEST(CudaVersionTest, BasicAssertions) {
#ifdef WITH_CUDA
  EXPECT_NE(pyg::cuda_version(), -1);
#else
  EXPECT_EQ(pyg::cuda_version(), -1);
#endif
}
