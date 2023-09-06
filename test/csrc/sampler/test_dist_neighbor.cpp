#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "pyg_lib/csrc/sampler/neighbor.h"
#include "pyg_lib/csrc/utils/types.h"
#include "test/csrc/graph.h"

TEST(FullDistNeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  int num_nodes = 6;
  auto graph = cycle_graph(num_nodes, options);
  auto seed = at::arange(2, 4, options);
  int64_t one_hop_num = -1;

  auto out = pyg::sampler::dist_neighbor_sample(
      /*rowptr=*/std::get<0>(graph), /*col=*/std::get<1>(graph), seed,
      one_hop_num);

  // sample nodes with duplicates
  auto expected_nodes = at::tensor({2, 3, 1, 3, 2, 4}, options);
  EXPECT_TRUE(at::equal(std::get<0>(out), expected_nodes));

  auto expected_edges = at::tensor({4, 5, 6, 7}, options);
  EXPECT_TRUE(at::equal(std::get<1>(out).value(), expected_edges));

  std::vector<int64_t> expected_cumm_sum_nbrs_per_node = {2, 4, 6};
  EXPECT_EQ(std::get<2>(out), expected_cumm_sum_nbrs_per_node);
}

TEST(WithoutReplacementNeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  int num_nodes = 6;
  auto graph = cycle_graph(num_nodes, options);
  auto seed = at::arange(2, 4, options);
  int64_t one_hop_num = 1;

  at::manual_seed(123456);
  auto out = pyg::sampler::dist_neighbor_sample(
      /*rowptr=*/std::get<0>(graph), /*col=*/std::get<1>(graph), seed,
      one_hop_num);

  // sample nodes with duplicates
  auto expected_nodes = at::tensor({2, 3, 1, 4}, options);
  EXPECT_TRUE(at::equal(std::get<0>(out), expected_nodes));

  auto expected_edges = at::tensor({4, 7}, options);
  EXPECT_TRUE(at::equal(std::get<1>(out).value(), expected_edges));

  std::vector<int64_t> expected_cumm_sum_nbrs_per_node = {2, 3, 4};
  EXPECT_EQ(std::get<2>(out), expected_cumm_sum_nbrs_per_node);
}

TEST(WithReplacementNeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  int num_nodes = 6;
  auto graph = cycle_graph(num_nodes, options);
  auto seed = at::arange(2, 4, options);
  int64_t one_hop_num = 2;

  at::manual_seed(123456);
  auto out = pyg::sampler::dist_neighbor_sample(
      /*rowptr=*/std::get<0>(graph), /*col=*/std::get<1>(graph), seed,
      one_hop_num, /*time=*/c10::nullopt,
      /*seed_time=*/c10::nullopt, /*edge_weight=*/c10::nullopt,
      /*csc*/ false, /*replace=*/true);

  // sample nodes with duplicates
  auto expected_nodes = at::tensor({2, 3, 1, 3, 4, 4}, options);
  EXPECT_TRUE(at::equal(std::get<0>(out), expected_nodes));

  auto expected_edges = at::tensor({4, 5, 7, 7}, options);
  EXPECT_TRUE(at::equal(std::get<1>(out).value(), expected_edges));

  std::vector<int64_t> expected_cumm_sum_nbrs_per_node = {2, 4, 6};
  EXPECT_EQ(std::get<2>(out), expected_cumm_sum_nbrs_per_node);
}

TEST(DistDisjointNeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  int num_nodes = 6;
  auto graph = cycle_graph(num_nodes, options);
  auto seed = at::arange(2, 4, options);
  int64_t one_hop_num = 2;
  //   auto batch = at::tensor({0, 1}, options);

  auto out = pyg::sampler::dist_neighbor_sample(
      /*rowptr=*/std::get<0>(graph), /*col=*/std::get<1>(graph), seed,
      one_hop_num, /*time=*/c10::nullopt,
      /*seed_time=*/c10::nullopt, /*edge_weight=*/c10::nullopt, /*csc*/ false,
      /*replace=*/false, /*directed=*/true, /*disjoint=*/true);

  // sample nodes with duplicates
  auto expected_nodes =
      at::tensor({0, 2, 1, 3, 0, 1, 0, 3, 1, 2, 1, 4}, options);
  EXPECT_TRUE(at::equal(std::get<0>(out), expected_nodes.view({-1, 2})));

  auto expected_edges = at::tensor({4, 5, 6, 7}, options);
  EXPECT_TRUE(at::equal(std::get<1>(out).value(), expected_edges));

  std::vector<int64_t> expected_cumm_sum_nbrs_per_node = {2, 4, 6};
  EXPECT_EQ(std::get<2>(out), expected_cumm_sum_nbrs_per_node);
}

TEST(DistTemporalNeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  int num_nodes = 6;
  auto graph = cycle_graph(num_nodes, options);
  auto rowptr = std::get<0>(graph);
  auto col = std::get<1>(graph);

  auto seed = at::arange(2, 4, options);
  int64_t one_hop_num = 2;

  // Time is equal to node ID ...
  auto time = at::arange(6, options);
  // ... so we need to sort the column vector by time/node ID:
  col = std::get<0>(at::sort(col.view({-1, 2}), /*dim=*/1)).flatten();

  auto out = pyg::sampler::dist_neighbor_sample(
      rowptr, col, seed, one_hop_num, time,
      /*seed_time=*/c10::nullopt, /*edge_weight=*/c10::nullopt,
      /*csc*/ false, /*replace=*/false, /*directed=*/true,
      /*disjoint=*/true, /*temporal_strategy=*/"uniform");

  // sample nodes with duplicates
  auto expected_nodes = at::tensor({0, 2, 1, 3, 0, 1, 1, 2}, options);
  EXPECT_TRUE(at::equal(std::get<0>(out), expected_nodes.view({-1, 2})));

  auto expected_edges = at::tensor({4, 6}, options);
  EXPECT_TRUE(at::equal(std::get<1>(out).value(), expected_edges));

  std::vector<int64_t> expected_cumm_sum_nbrs_per_node = {2, 3, 4};
  EXPECT_EQ(std::get<2>(out), expected_cumm_sum_nbrs_per_node);
}
