#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "pyg_lib/csrc/sampler/random_walk.h"
#include "test/csrc/graph.h"

TEST(RandomWalkTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);
#ifdef WITH_CUDA
  options = options.device(at::kCUDA);
#endif

  auto seed = at::arange(4, options);
  auto graph = cycle_graph(/*num_nodes=*/4, options);

  auto out = pyg::sampler::random_walk(/*rowptr=*/std::get<0>(graph),
                                       /*col=*/std::get<1>(graph), seed,
                                       /*walk_length=*/5);

  EXPECT_EQ(out.size(0), 4);
  EXPECT_EQ(out.size(1), 6);

  EXPECT_TRUE(at::equal(seed, out.select(/*dim=*/1, 0)));

  auto dist = (out.narrow(/*dim=*/1, 1, 5) - out.narrow(/*dim=*/1, 0, 5)).abs();
  EXPECT_TRUE(at::all((dist == 1) | (dist == 3)).item<bool>());
}
