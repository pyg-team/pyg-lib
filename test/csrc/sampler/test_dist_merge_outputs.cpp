#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "pyg_lib/csrc/sampler/dist_merge_outputs.h"
#include "pyg_lib/csrc/utils/types.h"

TEST(DistMergeOutputsTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  // seed = {0, 1, 2, 3}
  const std::vector<at::Tensor> node_ids = {
      at::tensor({2, 7, 8}, options),
      at::tensor({0, 1, 4, 5, 6}, options),
      at::tensor({3, 9, 10}, options),
  };
  const std::vector<at::Tensor> edge_ids = {
      at::tensor({17, 18}, options),
      at::tensor({14, 15, 16}, options),
      at::tensor({19, 20}, options),
  };

  const std::vector<std::vector<int64_t>> cumsum_neighbors_per_node = {
      {1, 3}, {2, 4, 5}, {1, 3}};
  const std::vector<int64_t> partition_ids = {1, 1, 0, 2};
  const std::vector<int64_t> partition_orders = {0, 1, 0, 0};

  auto out = pyg::sampler::merge_sampler_outputs(
      /*node_ids=*/node_ids,
      /*edge_ids=*/edge_ids,
      /*cumsum_neighbors_per_node=*/cumsum_neighbors_per_node,
      /*partition_ids=*/partition_ids,
      /*partition_orders=*/partition_orders,
      /*num_partitions=*/3,
      /*num_neighbors=*/2,
      /*batch=*/c10::nullopt,
      /*disjoint=*/false);

  auto expected_nodes = at::tensor({4, 5, 6, 7, 8, 9, 10}, options);
  EXPECT_TRUE(at::equal(std::get<0>(out), expected_nodes));

  auto expected_edges = at::tensor({14, 15, 16, 17, 18, 19, 20}, options);
  EXPECT_TRUE(at::equal(std::get<1>(out), expected_edges));

  const std::vector<int64_t> expected_num_sampled_neighbors_per_node = {2, 1, 2,
                                                                        2};
  EXPECT_EQ(std::get<3>(out), expected_num_sampled_neighbors_per_node);
}

TEST(DistMergeOutputsAllNeighborsTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  // seed = {0, 1, 2, 3}
  const std::vector<at::Tensor> node_ids = {
      at::tensor({2, 7, 8}, options),
      at::tensor({0, 1, 4, 5, 6}, options),
      at::tensor({3, 9, 10, 11}, options),
  };
  const std::vector<at::Tensor> edge_ids = {
      at::tensor({17, 18}, options),
      at::tensor({14, 15, 16}, options),
      at::tensor({19, 20, 21}, options),
  };

  const std::vector<std::vector<int64_t>> cumsum_neighbors_per_node = {
      {1, 3}, {2, 4, 5}, {1, 4}};
  const std::vector<int64_t> partition_ids = {1, 1, 0, 2};
  const std::vector<int64_t> partition_orders = {0, 1, 0, 0};

  auto out = pyg::sampler::merge_sampler_outputs(
      /*node_ids=*/node_ids,
      /*edge_ids=*/edge_ids,
      /*cumsum_neighbors_per_node=*/cumsum_neighbors_per_node,
      /*partition_ids=*/partition_ids,
      /*partition_orders=*/partition_orders,
      /*num_partitions=*/3,
      /*num_neighbors=*/-1,
      /*batch=*/c10::nullopt,
      /*disjoint=*/false);

  auto expected_nodes = at::tensor({4, 5, 6, 7, 8, 9, 10, 11}, options);
  EXPECT_TRUE(at::equal(std::get<0>(out), expected_nodes));

  auto expected_edges = at::tensor({14, 15, 16, 17, 18, 19, 20, 21}, options);
  EXPECT_TRUE(at::equal(std::get<1>(out), expected_edges));

  const std::vector<int64_t> expected_num_sampled_neighbors_per_node = {2, 1, 2,
                                                                        3};
  EXPECT_EQ(std::get<3>(out), expected_num_sampled_neighbors_per_node);
}

TEST(DistDisjointMergeOutputsTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  // seed = {0, 1, 2, 3}
  const std::vector<at::Tensor> node_ids = {
      at::tensor({2, 7, 8}, options),
      at::tensor({0, 1, 4, 5, 6}, options),
      at::tensor({3, 9, 10}, options),
  };
  const std::vector<at::Tensor> edge_ids = {
      at::tensor({17, 18}, options),
      at::tensor({14, 15, 16}, options),
      at::tensor({19, 20}, options),
  };
  const auto batch = at::tensor({0, 1, 2, 3}, options);

  const std::vector<std::vector<int64_t>> cumsum_neighbors_per_node = {
      {1, 3}, {2, 4, 5}, {1, 3}};
  const std::vector<int64_t> partition_ids = {1, 1, 0, 2};
  const std::vector<int64_t> partition_orders = {0, 1, 0, 0};

  auto out = pyg::sampler::merge_sampler_outputs(
      /*node_ids=*/node_ids,
      /*edge_ids=*/edge_ids,
      /*cumsum_neighbors_per_node=*/cumsum_neighbors_per_node,
      /*partition_ids=*/partition_ids,
      /*partition_orders=*/partition_orders,
      /*num_partitions=*/3,
      /*num_neighbors=*/2,
      /*batch=*/batch,
      /*disjoint=*/true);

  auto expected_nodes = at::tensor({4, 5, 6, 7, 8, 9, 10}, options);
  EXPECT_TRUE(at::equal(std::get<0>(out), expected_nodes));

  auto expected_batch = at::tensor({0, 0, 1, 2, 2, 3, 3}, options);
  EXPECT_TRUE(at::equal(std::get<2>(out).value(), expected_batch));

  const std::vector<int64_t> expected_num_sampled_neighbors_per_node = {2, 1, 2,
                                                                        2};
  EXPECT_EQ(std::get<3>(out), expected_num_sampled_neighbors_per_node);
}
