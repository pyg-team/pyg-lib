#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "pyg_lib/csrc/sampler/dist_relabel.h"
#include "pyg_lib/csrc/sampler/neighbor.h"
#include "pyg_lib/csrc/utils/types.h"
#include "test/csrc/graph.h"

TEST(DistRelabelNeighborhoodTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  auto seed = at::arange(2, 4, options);
  auto sampled_nodes_with_duplicates = at::tensor({1, 3, 2, 4}, options);
  std::vector<int64_t> num_sampled_neighbors_per_node = {2, 2};

  auto relabel_out = pyg::sampler::relabel_neighborhood(
      /*seed=*/seed,
      /*sampled_nodes_with_duplicates=*/sampled_nodes_with_duplicates,
      /*num_sampled_neighbors_per_node=*/num_sampled_neighbors_per_node,
      /*num_nodes=*/6);

  auto expected_row = at::tensor({0, 0, 1, 1}, options);
  EXPECT_TRUE(at::equal(std::get<0>(relabel_out), expected_row));
  auto expected_col = at::tensor({2, 1, 0, 3}, options);
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out), expected_col));

  // Check if output is correct:
  auto graph = cycle_graph(/*num_nodes=*/6, options);
  auto non_dist_out = pyg::sampler::neighbor_sample(
      /*rowptr=*/std::get<0>(graph),
      /*col=*/std::get<1>(graph),
      /*seed=*/seed,
      /*num_neighbors=*/{-1});

  EXPECT_TRUE(at::equal(std::get<0>(relabel_out), std::get<0>(non_dist_out)));
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out), std::get<1>(non_dist_out)));
}

TEST(DistDisjointRelabelNeighborhoodTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  auto seed = at::arange(2, 4, options);
  auto sampled_nodes_with_duplicates = at::tensor({1, 3, 2, 4}, options);
  std::vector<int64_t> num_sampled_neighbors_per_node = {2, 2};
  auto batch = at::tensor({0, 0, 1, 1}, options);

  auto relabel_out = pyg::sampler::relabel_neighborhood(
      /*seed=*/seed,
      /*sampled_nodes_with_duplicates=*/sampled_nodes_with_duplicates,
      /*num_sampled_neighbors_per_node=*/num_sampled_neighbors_per_node,
      /*num_nodes=*/6,
      /*batch=*/batch,
      /*csc=*/false,
      /*disjoint=*/true);

  auto expected_row = at::tensor({0, 0, 1, 1}, options);
  EXPECT_TRUE(at::equal(std::get<0>(relabel_out), expected_row));
  auto expected_col = at::tensor({2, 3, 4, 5}, options);
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out), expected_col));

  // Check if output is correct:
  auto graph = cycle_graph(/*num_nodes=*/6, options);
  auto non_dist_out = pyg::sampler::neighbor_sample(
      /*rowptr=*/std::get<0>(graph),
      /*col=*/std::get<1>(graph),
      /*seed=*/seed,
      /*num_neighbors=*/{2},
      /*node_time=*/c10::nullopt,
      /*edge_time=*/c10::nullopt,
      /*seed_time=*/c10::nullopt,
      /*edge_weight=*/c10::nullopt,
      /*csc*/ false,
      /*replace=*/false,
      /*directed=*/true,
      /*disjoint=*/true);

  EXPECT_TRUE(at::equal(std::get<0>(relabel_out), std::get<0>(non_dist_out)));
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out), std::get<1>(non_dist_out)));
}

TEST(DistHeteroRelabelNeighborhoodTest, BasicAssertions) {
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
  std::vector<int64_t> num_neighbors = {2};
  c10::Dict<rel_type, std::vector<int64_t>> num_neighbors_dict;
  num_neighbors_dict.insert(rel_key, num_neighbors);
  c10::Dict<node_type, int64_t> num_nodes_dict;
  num_nodes_dict.insert(node_key, 6);

  c10::Dict<node_type, at::Tensor> sampled_nodes_with_duplicates_dict;
  c10::Dict<rel_type, std::vector<std::vector<int64_t>>>
      num_sampled_neighbors_per_node_dict;
  sampled_nodes_with_duplicates_dict.insert(node_key,
                                            at::tensor({1, 3, 2, 4}, options));
  std::vector<std::vector<int64_t>> num_sampled_neighbors_per_node_vec(
      2, std::vector<int64_t>(1, 2));
  num_sampled_neighbors_per_node_dict.insert(
      rel_key, num_sampled_neighbors_per_node_vec);

  auto relabel_out = pyg::sampler::hetero_relabel_neighborhood(
      /*node_types=*/node_types,
      /*edge_types=*/edge_types,
      /*seed_dict=*/seed_dict,
      /*sampled_nodes_with_duplicates_dict=*/sampled_nodes_with_duplicates_dict,
      /*num_sampled_neighbors_per_node=*/num_sampled_neighbors_per_node_dict,
      /*num_nodes_dict=*/num_nodes_dict);

  auto expected_row = at::tensor({0, 0, 1, 1}, options);
  EXPECT_TRUE(at::equal(std::get<0>(relabel_out).at(rel_key), expected_row));
  auto expected_col = at::tensor({2, 1, 0, 3}, options);
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out).at(rel_key), expected_col));

  // Check if output is correct:
  auto non_dist_out = pyg::sampler::hetero_neighbor_sample(
      /*node_types=*/node_types,
      /*edge_types=*/edge_types,
      /*rowptr_dict=*/rowptr_dict,
      /*col_dict=*/col_dict,
      /*seed_dict=*/seed_dict,
      /*num_neighbors_dict=*/num_neighbors_dict);

  EXPECT_TRUE(at::equal(std::get<0>(relabel_out).at(rel_key),
                        std::get<0>(non_dist_out).at(rel_key)));
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out).at(rel_key),
                        std::get<1>(non_dist_out).at(rel_key)));
}

TEST(DistHeteroRelabelNeighborhoodCscTest, BasicAssertions) {
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
  std::vector<int64_t> num_neighbors = {2};
  c10::Dict<rel_type, std::vector<int64_t>> num_neighbors_dict;
  num_neighbors_dict.insert(rel_key, num_neighbors);
  c10::Dict<node_type, int64_t> num_nodes_dict;
  num_nodes_dict.insert(node_key, 6);

  c10::Dict<node_type, at::Tensor> sampled_nodes_with_duplicates_dict;
  c10::Dict<rel_type, std::vector<std::vector<int64_t>>>
      num_sampled_neighbors_per_node_dict;
  sampled_nodes_with_duplicates_dict.insert(node_key,
                                            at::tensor({1, 3, 2, 4}, options));
  std::vector<std::vector<int64_t>> num_sampled_neighbors_per_node_vec(
      2, std::vector<int64_t>(1, 2));
  num_sampled_neighbors_per_node_dict.insert(
      rel_key, num_sampled_neighbors_per_node_vec);

  auto relabel_out = pyg::sampler::hetero_relabel_neighborhood(
      /*node_types=*/node_types,
      /*edge_types=*/edge_types,
      /*seed_dict=*/seed_dict,
      /*sampled_nodes_with_duplicates_dict=*/sampled_nodes_with_duplicates_dict,
      /*num_sampled_neighbors_per_node=*/num_sampled_neighbors_per_node_dict,
      /*num_nodes_dict=*/num_nodes_dict,
      /*batch_dict=*/c10::nullopt,
      /*csc=*/true);

  auto expected_row = at::tensor({2, 1, 0, 3}, options);
  EXPECT_TRUE(at::equal(std::get<0>(relabel_out).at(rel_key), expected_row));
  auto expected_col = at::tensor({0, 0, 1, 1}, options);
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out).at(rel_key), expected_col));

  // Check if output is correct:
  auto non_dist_out = pyg::sampler::hetero_neighbor_sample(
      /*node_types=*/node_types,
      /*edge_types=*/edge_types,
      /*rowptr_dict=*/rowptr_dict,
      /*col_dict=*/col_dict,
      /*seed_dict=*/seed_dict,
      /*num_neighbors_dict=*/num_neighbors_dict,
      /*node_time_dict=*/c10::nullopt,
      /*edge_time_dict=*/c10::nullopt,
      /*seed_time_dict=*/c10::nullopt,
      /*edge_weight_dict=*/c10::nullopt,
      /*csc=*/true);

  EXPECT_TRUE(at::equal(std::get<0>(relabel_out).at(rel_key),
                        std::get<0>(non_dist_out).at(rel_key)));
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out).at(rel_key),
                        std::get<1>(non_dist_out).at(rel_key)));
}

TEST(DistHeteroDisjointRelabelNeighborhoodTest, BasicAssertions) {
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
  std::vector<int64_t> num_neighbors = {2};
  c10::Dict<rel_type, std::vector<int64_t>> num_neighbors_dict;
  num_neighbors_dict.insert(rel_key, num_neighbors);
  c10::Dict<node_type, int64_t> num_nodes_dict;
  num_nodes_dict.insert(node_key, 6);

  c10::Dict<node_type, at::Tensor> sampled_nodes_with_duplicates_dict;
  c10::Dict<rel_type, std::vector<std::vector<int64_t>>>
      num_sampled_neighbors_per_node_dict;
  c10::Dict<node_type, at::Tensor> batch_dict;
  sampled_nodes_with_duplicates_dict.insert(node_key,
                                            at::tensor({1, 3, 2, 4}, options));
  std::vector<std::vector<int64_t>> num_sampled_neighbors_per_node_vec(
      2, std::vector<int64_t>(1, 2));
  num_sampled_neighbors_per_node_dict.insert(
      rel_key, num_sampled_neighbors_per_node_vec);
  batch_dict.insert(node_key, at::tensor({0, 0, 1, 1}, options));

  auto relabel_out = pyg::sampler::hetero_relabel_neighborhood(
      /*node_types=*/node_types,
      /*edge_types=*/edge_types,
      /*seed_dict=*/seed_dict,
      /*sampled_nodes_with_duplicates_dict=*/sampled_nodes_with_duplicates_dict,
      /*num_sampled_neighbors_per_node=*/num_sampled_neighbors_per_node_dict,
      /*num_nodes_dict=*/num_nodes_dict,
      /*batch_dict=*/batch_dict,
      /*csc=*/false,
      /*disjoint=*/true);

  auto expected_row = at::tensor({0, 0, 1, 1}, options);
  EXPECT_TRUE(at::equal(std::get<0>(relabel_out).at(rel_key), expected_row));
  auto expected_col = at::tensor({2, 3, 4, 5}, options);
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out).at(rel_key), expected_col));

  // Check if output is correct:
  auto non_dist_out = pyg::sampler::hetero_neighbor_sample(
      /*node_types=*/node_types,
      /*edge_types=*/edge_types,
      /*rowptr_dict=*/rowptr_dict,
      /*col_dict=*/col_dict,
      /*seed_dict=*/seed_dict,
      /*num_neighbors_dict=*/num_neighbors_dict,
      /*node_time_dict=*/c10::nullopt,
      /*edge_time_dict=*/c10::nullopt,
      /*seed_time_dict=*/c10::nullopt,
      /*edge_weight_dict=*/c10::nullopt,
      /*csc=*/false,
      /*replace=*/false,
      /*directed=*/true,
      /*disjoint=*/true);

  EXPECT_TRUE(at::equal(std::get<0>(relabel_out).at(rel_key),
                        std::get<0>(non_dist_out).at(rel_key)));
  EXPECT_TRUE(at::equal(std::get<1>(relabel_out).at(rel_key),
                        std::get<1>(non_dist_out).at(rel_key)));
}
