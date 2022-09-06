#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "pyg_lib/csrc/sampler/neighbor.h"
#include "pyg_lib/csrc/utils/types.h"
#include "test/csrc/graph.h"

TEST(NeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  auto graph = cycle_graph(/*num_nodes=*/6, options);
  auto seed = at::arange(2, 4, options);
  std::vector<int64_t> num_neighbors = {2, 2};

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
}

TEST(DisjointNeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  auto graph = cycle_graph(/*num_nodes=*/6, options);
  auto seed = at::arange(2, 4, options);
  std::vector<int64_t> num_neighbors = {2, 2};

  auto out = pyg::sampler::neighbor_sample(
      /*rowptr=*/std::get<0>(graph),
      /*col=*/std::get<1>(graph), seed, num_neighbors, /*time=*/c10::nullopt,
      /*replace=*/false, /*directed=*/true, /*disjoint=*/true);

  auto expected_row = at::tensor({0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5}, options);
  EXPECT_TRUE(at::equal(std::get<0>(out), expected_row));
  auto expected_col = at::tensor({2, 3, 4, 5, 6, 0, 0, 7, 8, 1, 1, 9}, options);
  EXPECT_TRUE(at::equal(std::get<1>(out), expected_col));
  auto expected_nodes = at::tensor(
      {0, 2, 1, 3, 0, 1, 0, 3, 1, 2, 1, 4, 0, 0, 0, 4, 1, 1, 1, 5}, options);
  EXPECT_TRUE(at::equal(std::get<2>(out), expected_nodes.view({10, 2})));
  auto expected_edges =
      at::tensor({4, 5, 6, 7, 2, 3, 6, 7, 4, 5, 8, 9}, options);
  EXPECT_TRUE(at::equal(std::get<3>(out).value(), expected_edges));
}

TEST(HeteroNeighborTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  auto graph = cycle_graph(/*num_nodes=*/6, options);
  std::vector<node_t> node_types = {"paper"};
  std::vector<edge_t> edge_types = {{"paper", "to", "paper"}};
  c10::Dict<rel_t, at::Tensor> rowptr_dict;
  rowptr_dict.insert("paper__to__paper", std::get<0>(graph));
  c10::Dict<rel_t, at::Tensor> col_dict;
  col_dict.insert("paper__to__paper", std::get<1>(graph));
  c10::Dict<node_t, at::Tensor> seed_dict;
  seed_dict.insert("paper", at::arange(2, 4, options));
  std::vector<int64_t> num_neighbors = {2, 2};
  c10::Dict<rel_t, std::vector<int64_t>> num_neighbors_dict;
  num_neighbors_dict.insert("paper__to__paper", num_neighbors);

  auto out = pyg::sampler::hetero_neighbor_sample(
      node_types, edge_types, rowptr_dict, col_dict, seed_dict,
      num_neighbors_dict);
}
