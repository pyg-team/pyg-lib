#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "pyg_lib/csrc/partition/metis.h"
#include "test/csrc/graph.h"

TEST(MetisTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);
  auto graph = cycle_graph(/*num_nodes=*/6, options);

  auto out = pyg::partition::metis(std::get<0>(graph), std::get<1>(graph),
                                   /*num_partitions=*/2);
  EXPECT_EQ(out.numel(), 6);
  EXPECT_EQ(out.min().item<int64_t>(), 0);
  EXPECT_EQ(out.max().item<int64_t>(), 1);
}
