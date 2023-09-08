#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "pyg_lib/csrc/sampler/dist_merge_outputs.h"
#include "pyg_lib/csrc/utils/types.h"

TEST(DistMergeOutputsTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  auto partitions_num = 3;
  auto one_hop_num = 2;
  bool disjoint = false;
  bool with_edge = true;

  // seed = {0, 1, 2, 3}
  const std::vector<at::Tensor> nodes = {at::tensor({2, 7, 8}, options),
                                         at::tensor({0, 1, 4, 5, 6}, options),
                                         at::tensor({3, 9, 10}, options)};
  const std::vector<at::Tensor> edge_ids = {at::tensor({17, 18}, options),
                                            at::tensor({14, 15, 16}, options),
                                            at::tensor({19, 20}, options)};

  const std::vector<std::vector<int64_t>> cumm_sampled_nbrs_per_node = {
      {1, 3}, {2, 4, 5}, {1, 3}};
  const std::vector<int64_t> partition_ids = {1, 1, 0, 2};
  const std::vector<int64_t> partition_orders = {0, 1, 0, 0};

  auto out = pyg::sampler::merge_sampler_outputs(
      nodes, cumm_sampled_nbrs_per_node, partition_ids, partition_orders,
      partitions_num, one_hop_num, edge_ids, /*batch=*/c10::nullopt, disjoint,
      with_edge);

  auto expected_nodes = at::tensor({4, 5, 6, 7, 8, 9, 10}, options);
  EXPECT_TRUE(at::equal(std::get<0>(out), expected_nodes));

  auto expected_edges = at::tensor({14, 15, 16, 17, 18, 19, 20}, options);
  EXPECT_TRUE(at::equal(std::get<1>(out).value(), expected_edges));

  const std::vector<int64_t> expected_sampled_nbrs_per_node = {2, 1, 2, 2};
  EXPECT_EQ(std::get<3>(out), expected_sampled_nbrs_per_node);
}

TEST(DistMergeOutputsAllNeighborsTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  auto partitions_num = 3;
  auto one_hop_num = -1;
  bool disjoint = false;
  bool with_edge = true;

  // seed = {0, 1, 2, 3}
  const std::vector<at::Tensor> nodes = {at::tensor({2, 7, 8}, options),
                                         at::tensor({0, 1, 4, 5, 6}, options),
                                         at::tensor({3, 9, 10, 11}, options)};
  const std::vector<at::Tensor> edge_ids = {at::tensor({17, 18}, options),
                                            at::tensor({14, 15, 16}, options),
                                            at::tensor({19, 20, 21}, options)};

  const std::vector<std::vector<int64_t>> cumm_sampled_nbrs_per_node = {
      {1, 3}, {2, 4, 5}, {1, 4}};
  const std::vector<int64_t> partition_ids = {1, 1, 0, 2};
  const std::vector<int64_t> partition_orders = {0, 1, 0, 0};

  auto out = pyg::sampler::merge_sampler_outputs(
      nodes, cumm_sampled_nbrs_per_node, partition_ids, partition_orders,
      partitions_num, one_hop_num, edge_ids, /*batch=*/c10::nullopt, disjoint,
      with_edge);

  auto expected_nodes = at::tensor({4, 5, 6, 7, 8, 9, 10, 11}, options);
  EXPECT_TRUE(at::equal(std::get<0>(out), expected_nodes));

  auto expected_edges = at::tensor({14, 15, 16, 17, 18, 19, 20, 21}, options);
  EXPECT_TRUE(at::equal(std::get<1>(out).value(), expected_edges));

  const std::vector<int64_t> expected_sampled_nbrs_per_node = {2, 1, 2, 3};
  EXPECT_EQ(std::get<3>(out), expected_sampled_nbrs_per_node);
}

TEST(DistDisjointMergeOutputsTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  auto partitions_num = 3;
  auto one_hop_num = 2;
  bool disjoint = true;
  bool with_edge = false;

  // seed = {0, 1, 2, 3}
  const std::vector<at::Tensor> nodes = {at::tensor({2, 7, 8}, options),
                                         at::tensor({0, 1, 4, 5, 6}, options),
                                         at::tensor({3, 9, 10}, options)};
  const std::vector<at::Tensor> batch = {at::tensor({2, 2, 2}, options),
                                         at::tensor({0, 1, 0, 0, 1}, options),
                                         at::tensor({3, 3, 3}, options)};

  const std::vector<std::vector<int64_t>> cumm_sampled_nbrs_per_node = {
      {1, 3}, {2, 4, 5}, {1, 3}};
  const std::vector<int64_t> partition_ids = {1, 1, 0, 2};
  const std::vector<int64_t> partition_orders = {0, 1, 0, 0};

  auto out = pyg::sampler::merge_sampler_outputs(
      nodes, cumm_sampled_nbrs_per_node, partition_ids, partition_orders,
      partitions_num, one_hop_num, /*edge_ids=*/c10::nullopt, batch, disjoint,
      with_edge);

  auto expected_nodes = at::tensor({4, 5, 6, 7, 8, 9, 10}, options);
  EXPECT_TRUE(at::equal(std::get<0>(out), expected_nodes));

  auto expected_batch = at::tensor({0, 0, 1, 2, 2, 3, 3}, options);
  EXPECT_TRUE(at::equal(std::get<2>(out).value(), expected_batch));

  const std::vector<int64_t> expected_sampled_nbrs_per_node = {2, 1, 2, 2};
  EXPECT_EQ(std::get<3>(out), expected_sampled_nbrs_per_node);
}
