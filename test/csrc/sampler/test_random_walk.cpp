#include <gtest/gtest.h>

#include "../../../pyg_lib/csrc/sampler/random_walk.h"
#include "../graph.h"

TEST(RandomWalkTest, BasicAssertions) {
  auto options = torch::TensorOptions().dtype(torch::kInt64);
#ifdef WITH_CUDA
  options = options.device(torch::kCUDA);
#endif

  auto seed = torch::arange(4, options);
  auto graph = cycle_graph(/*num_nodes=*/4, options);

  auto out = pyg::sampler::random_walk(/*rowptr=*/std::get<0>(graph),
                                       /*col=*/std::get<1>(graph), seed,
                                       /*walk_length=*/5);

  EXPECT_EQ(out.size(0), 4);
  EXPECT_EQ(out.size(1), 6);

  EXPECT_TRUE(torch::all(seed == out.select(/*dim=*/1, 0)).item<bool>());

  auto dist = (out.narrow(/*dim=*/1, 1, 5) - out.narrow(/*dim=*/1, 0, 5)).abs();
  EXPECT_TRUE(torch::all((dist == 1) | (dist == 3)).item<bool>());
}
