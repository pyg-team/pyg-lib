#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "pyg_lib/csrc/sampler/subgraph.h"
#include "test/csrc/graph.h"

TEST(SubgraphTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  auto nodes = at::arange(1, 5, options);
  auto graph = cycle_graph(/*num_nodes=*/6, options);

  auto out = pyg::sampler::subgraph(/*rowptr=*/std::get<0>(graph),
                                    /*col=*/std::get<1>(graph), nodes);

  auto expected_rowptr = at::tensor({0, 1, 3, 5, 6}, options);
  EXPECT_TRUE(at::equal(std::get<0>(out), expected_rowptr));
  auto expected_col = at::tensor({1, 0, 2, 1, 3, 2}, options);
  EXPECT_TRUE(at::equal(std::get<1>(out), expected_col));
  auto expected_edge_id = at::tensor({3, 4, 5, 6, 7, 8}, options);
  EXPECT_TRUE(at::equal(std::get<2>(out).value(), expected_edge_id));
}
