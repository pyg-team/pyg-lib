#include <gtest/gtest.h>

#include "../../../pyg_lib/csrc/sampler/random_walk.h"
#include "../graph.h"

TEST(RandomWalkTest, BasicAssertions) {
  const auto graph = cycle_graph(/*num_nodes=*/4);
  const auto seed = torch::arange(4);

  auto out = pyg::sampler::random_walk(/*rowptr=*/std::get<0>(graph),
                                       /*col=*/std::get<1>(graph), seed,
                                       /*walk_length=*/5);

  EXPECT_EQ(out.size(0), 4);
  EXPECT_EQ(out.size(1), 6);

  auto dist = (out.narrow(/*dim=*/1, 1, 5) - out.narrow(/*dim=*/1, 0, 5)).abs();
  EXPECT_EQ(torch::all((dist == 1) | (dist == 3)).item<bool>(), true);
}
