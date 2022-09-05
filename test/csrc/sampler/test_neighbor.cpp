#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "pyg_lib/csrc/sampler/neighbor.h"
#include "test/csrc/graph.h"

TEST(NeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  auto graph = cycle_graph(/*num_nodes=*/6, options);
  auto seed = at::arange(2, 3, options);
  std::vector<int64_t> num_neighbors = {2, 2};

  auto out = pyg::sampler::neighbor_sample(/*rowptr=*/std::get<0>(graph),
                                           /*col=*/std::get<1>(graph), seed,
                                           num_neighbors);

  std::cout << std::get<0>(out) << std::endl;
  std::cout << std::get<1>(out) << std::endl;
  std::cout << std::get<2>(out) << std::endl;

  auto expected_row = at::tensor({0, 0, 1, 1, 2, 2}, options);
  EXPECT_TRUE(at::equal(std::get<0>(out), expected_row));
  auto expected_col = at::tensor({1, 2, 3, 0, 0, 4}, options);
  EXPECT_TRUE(at::equal(std::get<1>(out), expected_col));
  auto expected_nodes = at::tensor({2, 1, 3, 0, 4}, options);
  EXPECT_TRUE(at::equal(std::get<2>(out), expected_nodes));
  auto expected_edges = at::tensor({4, 5, 2, 3, 6, 7}, options);
  EXPECT_TRUE(at::equal(std::get<3>(out).value(), expected_edges));
}
