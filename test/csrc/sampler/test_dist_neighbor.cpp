#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "pyg_lib/csrc/sampler/dist_neighbor.h"
#include "pyg_lib/csrc/sampler/neighbor.h"
#include "pyg_lib/csrc/utils/types.h"
#include "test/csrc/graph.h"

TEST(FullDistNeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  int num_nodes = 6;
  auto graph = cycle_graph(num_nodes, options);
  auto seed = at::arange(2, 4, options);
  std::vector<int64_t> num_neighbors = {-1};

  auto out = pyg::sampler::neighbor_sample(
      /*rowptr=*/std::get<0>(graph),
      /*col=*/std::get<1>(graph), seed, num_neighbors,
      /*time=*/c10::nullopt, /*seed_time=*/c10::nullopt,
      /*batch=*/c10::nullopt, /*csc*/ false, /*replace=*/false,
      /*directed=*/true, /*disjoint=*/false,
      /*temporal_strategy=*/"uniform", /*return_edge_id=*/true,
      /*distributed=*/true);

  // do not sample rows and cols
  EXPECT_EQ(std::get<0>(out).numel(), 0);
  EXPECT_EQ(std::get<1>(out).numel(), 0);

  // sample nodes with duplicates
  auto expected_nodes = at::tensor({2, 3, 1, 3, 2, 4}, options);
  EXPECT_TRUE(at::equal(std::get<2>(out), expected_nodes));

  auto expected_edges = at::tensor({4, 5, 6, 7}, options);
  EXPECT_TRUE(at::equal(std::get<3>(out).value(), expected_edges));

  std::vector<int64_t> expected_cumm_sum_nbrs_per_node = {2, 4, 6};
  EXPECT_EQ(std::get<6>(out), expected_cumm_sum_nbrs_per_node);

  std::vector<int64_t> sampled_nbrs_per_node = {2, 2};
  // without seed nodes
  auto sampled_nodes_with_dupl = at::tensor({1, 3, 2, 4}, options);

  // get rows and cols
  auto relabel_out = pyg::sampler::relabel_neighborhood(
      seed, sampled_nodes_with_dupl, sampled_nbrs_per_node, num_nodes);

  auto expected_row = at::tensor({0, 0, 1, 1}, options);
  EXPECT_TRUE(at::equal(std::get<0>(relabel_out), expected_row));
  auto expected_col = at::tensor({2, 1, 0, 3}, options);
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out), expected_col));

  // check if rows and cols are the same as for the classic sampling
  auto non_dist_out = pyg::sampler::neighbor_sample(
      /*rowptr=*/std::get<0>(graph),
      /*col=*/std::get<1>(graph), seed, num_neighbors,
      /*time=*/c10::nullopt, /*seed_time=*/c10::nullopt,
      /*batch=*/c10::nullopt, /*csc*/ false, /*replace=*/false,
      /*directed=*/true, /*disjoint=*/false,
      /*temporal_strategy=*/"uniform",
      /*return_edge_id=*/true, /*distributed=*/false);

  EXPECT_TRUE(at::equal(std::get<0>(relabel_out), std::get<0>(non_dist_out)));
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out), std::get<1>(non_dist_out)));
}

TEST(WithoutReplacementNeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  int num_nodes = 6;
  auto graph = cycle_graph(num_nodes, options);
  auto seed = at::arange(2, 4, options);
  std::vector<int64_t> num_neighbors = {1};

  at::manual_seed(123456);
  auto out = pyg::sampler::neighbor_sample(
      /*rowptr=*/std::get<0>(graph),
      /*col=*/std::get<1>(graph), seed, num_neighbors,
      /*time=*/c10::nullopt, /*seed_time=*/c10::nullopt,
      /*batch=*/c10::nullopt, /*csc*/ false, /*replace=*/false,
      /*directed=*/true, /*disjoint=*/false,
      /*temporal_strategy=*/"uniform", /*return_edge_id=*/true,
      /*distributed=*/true);

  // do not sample rows and cols
  EXPECT_EQ(std::get<0>(out).numel(), 0);
  EXPECT_EQ(std::get<1>(out).numel(), 0);

  // sample nodes with duplicates
  auto expected_nodes = at::tensor({2, 3, 1, 4}, options);
  EXPECT_TRUE(at::equal(std::get<2>(out), expected_nodes));

  auto expected_edges = at::tensor({4, 7}, options);
  EXPECT_TRUE(at::equal(std::get<3>(out).value(), expected_edges));

  std::vector<int64_t> expected_cumm_sum_nbrs_per_node = {2, 3, 4};
  EXPECT_EQ(std::get<6>(out), expected_cumm_sum_nbrs_per_node);

  std::vector<int64_t> sampled_nbrs_per_node = {1, 1};
  // without seed nodes
  auto sampled_nodes_with_dupl = at::tensor({1, 4}, options);

  // get rows and cols
  auto relabel_out = pyg::sampler::relabel_neighborhood(
      seed, sampled_nodes_with_dupl, sampled_nbrs_per_node, num_nodes);

  auto expected_row = at::tensor({0, 1}, options);
  EXPECT_TRUE(at::equal(std::get<0>(relabel_out), expected_row));
  auto expected_col = at::tensor({2, 3}, options);
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out), expected_col));
}

TEST(WithReplacementNeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  int num_nodes = 6;
  auto graph = cycle_graph(num_nodes, options);
  auto seed = at::arange(2, 4, options);
  std::vector<int64_t> num_neighbors = {2};

  at::manual_seed(123456);
  auto out = pyg::sampler::neighbor_sample(
      /*rowptr=*/std::get<0>(graph), /*col=*/std::get<1>(graph), seed,
      num_neighbors, /*time=*/c10::nullopt,
      /*seed_time=*/c10::nullopt, /*batch=*/c10::nullopt,
      /*csc*/ false, /*replace=*/true, /*directed=*/true,
      /*disjoint=*/false, /*temporal_strategy=*/"uniform",
      /*return_edge_id=*/true, /*distributed=*/true);

  // do not sample rows and cols
  EXPECT_EQ(std::get<0>(out).numel(), 0);
  EXPECT_EQ(std::get<1>(out).numel(), 0);

  // sample nodes with duplicates
  auto expected_nodes = at::tensor({2, 3, 1, 3, 4, 4}, options);
  EXPECT_TRUE(at::equal(std::get<2>(out), expected_nodes));

  auto expected_edges = at::tensor({4, 5, 7, 7}, options);
  EXPECT_TRUE(at::equal(std::get<3>(out).value(), expected_edges));

  std::vector<int64_t> expected_cumm_sum_nbrs_per_node = {2, 4, 6};
  EXPECT_EQ(std::get<6>(out), expected_cumm_sum_nbrs_per_node);

  std::vector<int64_t> sampled_nbrs_per_node = {2, 2};
  // without seed nodes
  auto sampled_nodes_with_dupl = at::tensor({1, 3, 4, 4}, options);

  // get rows and cols
  auto relabel_out = pyg::sampler::relabel_neighborhood(
      seed, sampled_nodes_with_dupl, sampled_nbrs_per_node, num_nodes);

  auto expected_row = at::tensor({0, 0, 1, 1}, options);
  EXPECT_TRUE(at::equal(std::get<0>(relabel_out), expected_row));
  auto expected_col = at::tensor({2, 1, 3, 3}, options);
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out), expected_col));
}

TEST(DistDisjointNeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  int num_nodes = 6;
  auto graph = cycle_graph(num_nodes, options);
  auto seed = at::arange(2, 4, options);
  std::vector<int64_t> num_neighbors = {2};
  auto batch = at::tensor({0, 1}, options);

  auto out = pyg::sampler::neighbor_sample(
      /*rowptr=*/std::get<0>(graph), /*col=*/std::get<1>(graph), seed,
      num_neighbors, /*time=*/c10::nullopt,
      /*seed_time=*/c10::nullopt, batch, /*csc*/ false,
      /*replace=*/false, /*directed=*/true, /*disjoint=*/true,
      /*temporal_strategy=*/"uniform", /*return_edge_id=*/true,
      /*distributed=*/true);

  // do not sample rows and cols
  EXPECT_EQ(std::get<0>(out).numel(), 0);
  EXPECT_EQ(std::get<1>(out).numel(), 0);

  // sample nodes with duplicates
  auto expected_nodes =
      at::tensor({0, 2, 1, 3, 0, 1, 0, 3, 1, 2, 1, 4}, options);
  EXPECT_TRUE(at::equal(std::get<2>(out), expected_nodes.view({-1, 2})));

  auto expected_edges = at::tensor({4, 5, 6, 7}, options);
  EXPECT_TRUE(at::equal(std::get<3>(out).value(), expected_edges));

  std::vector<int64_t> expected_cumm_sum_nbrs_per_node = {2, 4, 6};
  EXPECT_EQ(std::get<6>(out), expected_cumm_sum_nbrs_per_node);

  std::vector<int64_t> sampled_nbrs_per_node = {2, 2};
  // without seed nodes
  auto sampled_nodes_with_dupl = at::tensor({1, 3, 2, 4}, options);
  auto sampled_batch = at::tensor({0, 0, 1, 1}, options);

  // get rows and cols
  auto relabel_out = pyg::sampler::relabel_neighborhood(
      seed, sampled_nodes_with_dupl, sampled_nbrs_per_node, num_nodes,
      sampled_batch, /*csc=*/false, /*disjoint=*/true);

  auto expected_row = at::tensor({0, 0, 1, 1}, options);
  EXPECT_TRUE(at::equal(std::get<0>(relabel_out), expected_row));
  auto expected_col = at::tensor({2, 3, 4, 5}, options);
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out), expected_col));

  // check if rows and cols are the same as for the classic sampling
  auto non_dist_out = pyg::sampler::neighbor_sample(
      /*rowptr=*/std::get<0>(graph),
      /*col=*/std::get<1>(graph), seed, num_neighbors,
      /*time=*/c10::nullopt, /*seed_time=*/c10::nullopt, batch, /*csc*/ false,
      /*replace=*/false,
      /*directed=*/true, /*disjoint=*/true,
      /*temporal_strategy=*/"uniform",
      /*return_edge_id=*/true, /*distributed=*/false);

  EXPECT_TRUE(at::equal(std::get<0>(relabel_out), std::get<0>(non_dist_out)));
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out), std::get<1>(non_dist_out)));
}

TEST(DistTemporalNeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  int num_nodes = 6;
  auto graph = cycle_graph(num_nodes, options);
  auto rowptr = std::get<0>(graph);
  auto col = std::get<1>(graph);

  auto seed = at::arange(2, 4, options);
  std::vector<int64_t> num_neighbors = {2};

  // Time is equal to node ID ...
  auto time = at::arange(6, options);
  // ... so we need to sort the column vector by time/node ID:
  col = std::get<0>(at::sort(col.view({-1, 2}), /*dim=*/1)).flatten();

  auto out = pyg::sampler::neighbor_sample(
      rowptr, col, seed, num_neighbors, time,
      /*seed_time=*/c10::nullopt, /*batch=*/c10::nullopt,
      /*csc*/ false, /*replace=*/false, /*directed=*/true,
      /*disjoint=*/true, /*temporal_strategy=*/"uniform",
      /*return_edge_id=*/true, /*distributed=*/true);

  // do not sample rows and cols
  EXPECT_EQ(std::get<0>(out).numel(), 0);
  EXPECT_EQ(std::get<1>(out).numel(), 0);

  // sample nodes with duplicates
  auto expected_nodes = at::tensor({0, 2, 1, 3, 0, 1, 1, 2}, options);
  EXPECT_TRUE(at::equal(std::get<2>(out), expected_nodes.view({-1, 2})));

  auto expected_edges = at::tensor({4, 6}, options);
  EXPECT_TRUE(at::equal(std::get<3>(out).value(), expected_edges));

  std::vector<int64_t> expected_cumm_sum_nbrs_per_node = {2, 3, 4};
  EXPECT_EQ(std::get<6>(out), expected_cumm_sum_nbrs_per_node);

  std::vector<int64_t> sampled_nbrs_per_node = {1, 1};
  // without seed nodes
  auto sampled_nodes_with_dupl = at::tensor({1, 2}, options);
  auto sampled_batch = at::tensor({0, 1}, options);

  // get rows and cols
  auto relabel_out = pyg::sampler::relabel_neighborhood(
      seed, sampled_nodes_with_dupl, sampled_nbrs_per_node, num_nodes,
      sampled_batch, /*csc=*/false, /*disjoint=*/true);

  auto expected_row = at::tensor({0, 1}, options);
  EXPECT_TRUE(at::equal(std::get<0>(relabel_out), expected_row));
  auto expected_col = at::tensor({2, 3}, options);
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out), expected_col));

  // check if rows and cols are the same as for the classic sampling
  auto non_dist_out = pyg::sampler::neighbor_sample(
      rowptr, col, seed, num_neighbors, time, /*seed_time=*/c10::nullopt,
      /*batch=*/c10::nullopt, /*csc*/ false,
      /*replace=*/false, /*directed=*/true,
      /*disjoint=*/true, /*temporal_strategy=*/"uniform",
      /*return_edge_id=*/true, /*distributed=*/false);

  EXPECT_TRUE(at::equal(std::get<0>(relabel_out), std::get<0>(non_dist_out)));
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out), std::get<1>(non_dist_out)));
}

TEST(DistHeteroNeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  int num_nodes = 6;
  auto graph = cycle_graph(num_nodes, options);
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
  std::vector<int64_t> num_neighbors = {2};
  c10::Dict<rel_type, std::vector<int64_t>> num_neighbors_dict;
  num_neighbors_dict.insert(rel_key, num_neighbors);
  c10::Dict<node_type, int64_t> num_nodes_dict;
  num_nodes_dict.insert(node_key, num_nodes);

  c10::Dict<node_type, at::Tensor> sampled_nodes_with_dupl_dict;
  c10::Dict<rel_type, std::vector<int64_t>> sampled_nbrs_per_node_dict;
  sampled_nodes_with_dupl_dict.insert(node_key,
                                      at::tensor({1, 3, 2, 4}, options));
  sampled_nbrs_per_node_dict.insert(rel_key, std::vector<int64_t>(2, 2));
  // get rows and cols
  auto relabel_out = pyg::sampler::hetero_relabel_neighborhood(
      node_types, edge_types, seed_dict, sampled_nodes_with_dupl_dict,
      sampled_nbrs_per_node_dict, num_nodes_dict,
      /*batch_dict=*/c10::nullopt, /*csc=*/false, /*disjoint=*/false);

  auto expected_row = at::tensor({0, 0, 1, 1}, options);
  EXPECT_TRUE(at::equal(std::get<0>(relabel_out).at(rel_key), expected_row));
  auto expected_col = at::tensor({2, 1, 0, 3}, options);
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out).at(rel_key), expected_col));

  // check if rows and cols are the same as for the classic sampling
  auto non_dist_out = pyg::sampler::hetero_neighbor_sample(
      node_types, edge_types, rowptr_dict, col_dict, seed_dict,
      num_neighbors_dict);

  EXPECT_TRUE(at::equal(std::get<0>(relabel_out).at(rel_key),
                        std::get<0>(non_dist_out).at(rel_key)));
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out).at(rel_key),
                        std::get<1>(non_dist_out).at(rel_key)));
}

TEST(DistHeteroCscNeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  int num_nodes = 6;
  auto graph = cycle_graph(num_nodes, options);
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
  std::vector<int64_t> num_neighbors = {2};
  c10::Dict<rel_type, std::vector<int64_t>> num_neighbors_dict;
  num_neighbors_dict.insert(rel_key, num_neighbors);
  c10::Dict<node_type, int64_t> num_nodes_dict;
  num_nodes_dict.insert(node_key, num_nodes);

  c10::Dict<node_type, at::Tensor> sampled_nodes_with_dupl_dict;
  c10::Dict<rel_type, std::vector<int64_t>> sampled_nbrs_per_node_dict;
  sampled_nodes_with_dupl_dict.insert(node_key,
                                      at::tensor({1, 3, 2, 4}, options));
  sampled_nbrs_per_node_dict.insert(rel_key, std::vector<int64_t>(2, 2));
  // get rows and cols
  auto relabel_out = pyg::sampler::hetero_relabel_neighborhood(
      node_types, edge_types, seed_dict, sampled_nodes_with_dupl_dict,
      sampled_nbrs_per_node_dict, num_nodes_dict,
      /*batch_dict=*/c10::nullopt, /*csc=*/true, /*disjoint=*/false);

  auto expected_row = at::tensor({2, 1, 0, 3}, options);
  EXPECT_TRUE(at::equal(std::get<0>(relabel_out).at(rel_key), expected_row));
  auto expected_col = at::tensor({0, 0, 1, 1}, options);
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out).at(rel_key), expected_col));

  // check if rows and cols are the same as for the classic sampling
  auto non_dist_out = pyg::sampler::hetero_neighbor_sample(
      node_types, edge_types, rowptr_dict, col_dict, seed_dict,
      num_neighbors_dict,
      /*time_dict=*/c10::nullopt,
      /*seed_time_dict=*/c10::nullopt, /*csc*/ true);

  EXPECT_TRUE(at::equal(std::get<0>(relabel_out).at(rel_key),
                        std::get<0>(non_dist_out).at(rel_key)));
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out).at(rel_key),
                        std::get<1>(non_dist_out).at(rel_key)));
}

TEST(DistHeteroDisjointNeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  int num_nodes = 6;
  auto graph = cycle_graph(num_nodes, options);
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
  std::vector<int64_t> num_neighbors = {2};
  c10::Dict<rel_type, std::vector<int64_t>> num_neighbors_dict;
  num_neighbors_dict.insert(rel_key, num_neighbors);
  c10::Dict<node_type, int64_t> num_nodes_dict;
  num_nodes_dict.insert(node_key, num_nodes);

  c10::Dict<node_type, at::Tensor> sampled_nodes_with_dupl_dict;
  c10::Dict<rel_type, std::vector<int64_t>> sampled_nbrs_per_node_dict;
  c10::Dict<node_type, at::Tensor> batch_dict;
  sampled_nodes_with_dupl_dict.insert(node_key,
                                      at::tensor({1, 3, 2, 4}, options));
  sampled_nbrs_per_node_dict.insert(rel_key, std::vector<int64_t>(2, 2));
  batch_dict.insert(node_key, at::tensor({0, 0, 1, 1}, options));
  // get rows and cols
  auto relabel_out = pyg::sampler::hetero_relabel_neighborhood(
      node_types, edge_types, seed_dict, sampled_nodes_with_dupl_dict,
      sampled_nbrs_per_node_dict, num_nodes_dict, batch_dict,
      /*csc=*/false, /*disjoint=*/true);

  auto expected_row = at::tensor({0, 0, 1, 1}, options);
  EXPECT_TRUE(at::equal(std::get<0>(relabel_out).at(rel_key), expected_row));
  auto expected_col = at::tensor({2, 3, 4, 5}, options);
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out).at(rel_key), expected_col));

  // check if rows and cols are the same as for the classic sampling
  auto non_dist_out = pyg::sampler::hetero_neighbor_sample(
      node_types, edge_types, rowptr_dict, col_dict, seed_dict,
      num_neighbors_dict, /*time_dict=*/c10::nullopt,
      /*seed_time_dict=*/c10::nullopt, /*csc=*/false, /*replace=*/false,
      /*directed=*/true, /*disjoint=*/true);

  EXPECT_TRUE(at::equal(std::get<0>(relabel_out).at(rel_key),
                        std::get<0>(non_dist_out).at(rel_key)));
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out).at(rel_key),
                        std::get<1>(non_dist_out).at(rel_key)));
}
