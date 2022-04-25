#include <gtest/gtest.h>

#include "../../../pyg_lib/csrc/sampler/random_walk.h"

TEST(RandomWalkTest, BasicAssertions) {
  const auto options = torch::TensorOptions().dtype(torch::kInt64);
  const auto rowptr = torch::tensor({0, 2, 4, 6, 8}, options);
  const auto col = torch::tensor({1, 3, 0, 2, 1, 3, 2, 0}, options);
  const auto seed = torch::arange(4, options);

  /* std::cout << rowptr << std::endl; */
  /* std::cout << col << std::endl; */
  /* std::cout << seed << std::endl; */

  auto out = pyg::sampler::random_walk(rowptr, col, seed, /*walk_length=*/3);

  /* row = tensor([ 0, 1, 1, 1, 2, 2, 3, 3, 4, 4 ], torch.long, device) col = */
  /*     tensor([ 1, 0, 2, 3, 1, 4, 1, 4, 2, 3 ], torch.long, device) */

  /* torch::Tensor */
}
