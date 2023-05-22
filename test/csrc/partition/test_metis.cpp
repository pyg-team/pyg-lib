#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "pyg_lib/csrc/partition/metis.h"
#include "test/csrc/graph.h"

TEST(MetisTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  auto graph = cycle_graph(/*num_nodes=*/6, options);
  std::cout << std::get<0>(graph) << std::endl;
  std::cout << std::get<1>(graph) << std::endl;

  auto out = pyg::partition::metis(std::get<0>(graph), std::get<1>(graph),
                                   /*num_partitions=*/2);
  std::cout << out << std::endl;
}
