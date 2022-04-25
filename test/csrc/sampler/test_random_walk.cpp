#include <gtest/gtest.h>

#include "../../../pyg_lib/csrc/sampler/random_walk.h"

TEST(RandomWalkTest, BasicAssertions) {
  const auto options = torch::TensorOptions().dtype(torch::kInt64);
  const auto rowptr = torch::tensor({0, 2, 4, 6, 8}, options);
  const auto col = torch::tensor({1, 3, 0, 2, 1, 3, 2, 0}, options);
  const auto seed = torch::arange(4, options);

  auto out = pyg::sampler::random_walk(rowptr, col, seed, /*walk_length=*/5);

  EXPECT_EQ(out.size(0), 4);
  EXPECT_EQ(out.size(1), 6);

  auto dist = (out.narrow(/*dim=*/1, 1, 5) - out.narrow(/*dim=*/1, 0, 5)).abs();
  EXPECT_EQ(torch::all((dist == 1) | (dist == 3)).item<bool>(), true);
}
