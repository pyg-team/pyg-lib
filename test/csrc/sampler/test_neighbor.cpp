#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "pyg_lib/csrc/sampler/neighbor.h"
#include "pyg_lib/csrc/utils/types.h"
#include "test/csrc/graph.h"

TEST(FullNeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  auto graph = cycle_graph(/*num_nodes=*/6, options);
  auto seed = at::arange(2, 4, options);
  std::vector<int64_t> num_neighbors = {-1, -1};

  auto out = pyg::sampler::neighbor_sample(/*rowptr=*/std::get<0>(graph),
                                           /*col=*/std::get<1>(graph), seed,
                                           num_neighbors);

  auto expected_row = at::tensor({0, 0, 1, 1, 2, 2, 3, 3}, options);
  EXPECT_TRUE(at::equal(std::get<0>(out), expected_row));
  auto expected_col = at::tensor({2, 1, 0, 3, 4, 0, 1, 5}, options);
  EXPECT_TRUE(at::equal(std::get<1>(out), expected_col));
  auto expected_nodes = at::tensor({2, 3, 1, 4, 0, 5}, options);
  EXPECT_TRUE(at::equal(std::get<2>(out), expected_nodes));
  auto expected_edges = at::tensor({4, 5, 6, 7, 2, 3, 8, 9}, options);
  EXPECT_TRUE(at::equal(std::get<3>(out).value(), expected_edges));
  std::vector<int64_t> expected_num_nodes = {2, 2, 2};
  EXPECT_TRUE(std::get<4>(out) == expected_num_nodes);
  std::vector<int64_t> expected_num_edges = {4, 4};
  EXPECT_TRUE(std::get<5>(out) == expected_num_edges);
}

TEST(WithoutReplacementNeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  auto graph = cycle_graph(/*num_nodes=*/6, options);
  auto seed = at::arange(2, 4, options);
  std::vector<int64_t> num_neighbors = {1, 1};

  at::manual_seed(123456);
  auto out = pyg::sampler::neighbor_sample(
      /*rowptr=*/std::get<0>(graph),
      /*col=*/std::get<1>(graph), seed, num_neighbors, /*time=*/c10::nullopt,
      /*seed_time=*/c10::nullopt, /*csc=*/false, /*replace=*/false);

  auto expected_row = at::tensor({0, 1, 2, 3}, options);
  EXPECT_TRUE(at::equal(std::get<0>(out), expected_row));
  auto expected_col = at::tensor({2, 3, 0, 4}, options);
  EXPECT_TRUE(at::equal(std::get<1>(out), expected_col));
  auto expected_nodes = at::tensor({2, 3, 1, 4, 5}, options);
  EXPECT_TRUE(at::equal(std::get<2>(out), expected_nodes));
  auto expected_edges = at::tensor({4, 7, 3, 9}, options);
  EXPECT_TRUE(at::equal(std::get<3>(out).value(), expected_edges));
}

TEST(WithReplacementNeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  auto graph = cycle_graph(/*num_nodes=*/6, options);
  auto seed = at::arange(2, 4, options);
  std::vector<int64_t> num_neighbors = {1, 1};

  at::manual_seed(123456);
  auto out = pyg::sampler::neighbor_sample(
      /*rowptr=*/std::get<0>(graph),
      /*col=*/std::get<1>(graph), seed, num_neighbors, /*time=*/c10::nullopt,
      /*seed_time=*/c10::nullopt, /*csc=*/false, /*replace=*/true);

  auto expected_row = at::tensor({0, 1, 2, 3}, options);
  EXPECT_TRUE(at::equal(std::get<0>(out), expected_row));
  auto expected_col = at::tensor({2, 3, 0, 4}, options);
  EXPECT_TRUE(at::equal(std::get<1>(out), expected_col));
  auto expected_nodes = at::tensor({2, 3, 1, 4, 5}, options);
  EXPECT_TRUE(at::equal(std::get<2>(out), expected_nodes));
  auto expected_edges = at::tensor({4, 7, 3, 9}, options);
  EXPECT_TRUE(at::equal(std::get<3>(out).value(), expected_edges));
}

TEST(DisjointNeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  auto graph = cycle_graph(/*num_nodes=*/6, options);
  auto seed = at::arange(2, 4, options);
  std::vector<int64_t> num_neighbors = {2, 2};

  auto out = pyg::sampler::neighbor_sample(
      /*rowptr=*/std::get<0>(graph),
      /*col=*/std::get<1>(graph), seed, num_neighbors, /*time=*/c10::nullopt,
      /*seed_time=*/c10::nullopt, /*csc=*/false, /*replace=*/false,
      /*directed=*/true, /*disjoint=*/true);

  auto expected_row = at::tensor({0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5}, options);
  EXPECT_TRUE(at::equal(std::get<0>(out), expected_row));
  auto expected_col = at::tensor({2, 3, 4, 5, 6, 0, 0, 7, 8, 1, 1, 9}, options);
  EXPECT_TRUE(at::equal(std::get<1>(out), expected_col));
  auto expected_nodes = at::tensor(
      {0, 2, 1, 3, 0, 1, 0, 3, 1, 2, 1, 4, 0, 0, 0, 4, 1, 1, 1, 5}, options);
  EXPECT_TRUE(at::equal(std::get<2>(out), expected_nodes.view({-1, 2})));
  auto expected_edges =
      at::tensor({4, 5, 6, 7, 2, 3, 6, 7, 4, 5, 8, 9}, options);
  EXPECT_TRUE(at::equal(std::get<3>(out).value(), expected_edges));
}

TEST(TemporalNeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  auto graph = cycle_graph(/*num_nodes=*/6, options);
  auto rowptr = std::get<0>(graph);
  auto col = std::get<1>(graph);
  auto seed = at::arange(2, 4, options);

  // Time is equal to node ID ...
  auto time = at::arange(6, options);
  // ... so we need to sort the column vector by time/node ID:
  col = std::get<0>(at::sort(col.view({-1, 2}), /*dim=*/1)).flatten();

  auto out1 = pyg::sampler::neighbor_sample(
      rowptr, col, seed, /*num_neighbors=*/{2, 2}, /*time=*/time,
      /*seed_time=*/c10::nullopt, /*csc=*/false, /*replace=*/false,
      /*directed=*/true, /*disjoint=*/true);

  // Expect only the earlier neighbors or the same node to be sampled:
  auto expected_row = at::tensor({0, 1, 2, 2, 3, 3}, options);
  EXPECT_TRUE(at::equal(std::get<0>(out1), expected_row));
  auto expected_col = at::tensor({2, 3, 4, 0, 5, 1}, options);
  EXPECT_TRUE(at::equal(std::get<1>(out1), expected_col));
  auto expected_nodes =
      at::tensor({0, 2, 1, 3, 0, 1, 1, 2, 0, 0, 1, 1}, options);
  EXPECT_TRUE(at::equal(std::get<2>(out1), expected_nodes.view({-1, 2})));
  auto expected_edges = at::tensor({4, 6, 2, 3, 4, 5}, options);
  EXPECT_TRUE(at::equal(std::get<3>(out1).value(), expected_edges));

  auto out2 = pyg::sampler::neighbor_sample(
      rowptr, col, seed, /*num_neighbors=*/{1, 2}, /*time=*/time,
      /*seed_time=*/c10::nullopt, /*csc=*/false, /*replace=*/false,
      /*directed=*/true, /*disjoint=*/true, /*temporal_strategy=*/"last");

  EXPECT_TRUE(at::equal(std::get<0>(out1), std::get<0>(out2)));
  EXPECT_TRUE(at::equal(std::get<1>(out1), std::get<1>(out2)));
  EXPECT_TRUE(at::equal(std::get<2>(out1), std::get<2>(out2)));
  EXPECT_TRUE(at::equal(std::get<3>(out1).value(), std::get<3>(out2).value()));
}

TEST(HeteroNeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  auto graph = cycle_graph(/*num_nodes=*/6, options);
  const auto node_key = "paper";
  const auto edge_key = std::make_tuple("paper", "to", "paper");
  const auto rel_key = "paper__to__paper";
  std::vector<node_type> node_types = {node_key};
  std::vector<edge_type> edge_types = {edge_key};
  c10::Dict<rel_type, at::Tensor> rowptr_dict;
  rowptr_dict.insert(rel_key, std::get<0>(graph));
  c10::Dict<rel_type, at::Tensor> col_dict;
  col_dict.insert(rel_key, std::get<1>(graph));
  c10::Dict<node_type, at::Tensor> seed_dict;
  seed_dict.insert(node_key, at::arange(2, 4, options));
  std::vector<int64_t> num_neighbors = {2, 2};
  c10::Dict<rel_type, std::vector<int64_t>> num_neighbors_dict;
  num_neighbors_dict.insert(rel_key, num_neighbors);

  auto out = pyg::sampler::hetero_neighbor_sample(
      node_types, edge_types, rowptr_dict, col_dict, seed_dict,
      num_neighbors_dict);

  auto expected_row = at::tensor({0, 0, 1, 1, 2, 2, 3, 3}, options);
  EXPECT_TRUE(at::equal(std::get<0>(out).at(rel_key), expected_row));
  auto expected_col = at::tensor({2, 1, 0, 3, 4, 0, 1, 5}, options);
  EXPECT_TRUE(at::equal(std::get<1>(out).at(rel_key), expected_col));
  auto expected_nodes = at::tensor({2, 3, 1, 4, 0, 5}, options);
  EXPECT_TRUE(at::equal(std::get<2>(out).at(node_key), expected_nodes));
  auto expected_edges = at::tensor({4, 5, 6, 7, 2, 3, 8, 9}, options);
  EXPECT_TRUE(at::equal(std::get<3>(out).value().at(rel_key), expected_edges));
  std::vector<int64_t> expected_num_nodes = {2, 2, 2};
  EXPECT_TRUE(std::get<4>(out).at("paper") == expected_num_nodes);
  std::vector<int64_t> expected_num_edges = {4, 4};
  EXPECT_TRUE(std::get<5>(out).at("paper__to__paper") == expected_num_edges);
}
