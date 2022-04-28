#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "pyg_lib/csrc/sampler/subgraph.h"
#include "test/csrc/graph.h"

TEST(SubgraphTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  auto nodes = at::arange(4, options);
  auto graph = cycle_graph(/*num_nodes=*/6, options);

  auto out = pyg::sampler::subgraph(/*rowptr=*/std::get<0>(graph),
                                    /*col=*/std::get<1>(graph), nodes);

  std::cout << std::get<0>(out) << std::endl;
  std::cout << std::get<1>(out) << std::endl;
  std::cout << std::get<2>(out) << std::endl;
}
