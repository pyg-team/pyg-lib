#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "pyg_lib/csrc/classes/cpu/neighbor_sampler.h"
#include "pyg_lib/csrc/utils/types.h"

TEST(NeighborSamplerTest, BasicAssertions) {
  const std::vector<edge_type> edge_types({{"A", "to", "B"}, {"B", "to", "C"}});
  c10::Dict<rel_type, std::vector<int64_t>> num_neighbors;
  num_neighbors.insert("A__to__B", std::vector<int64_t>({10, 0}));
  num_neighbors.insert("B__to__C", std::vector<int64_t>({0, 2}));
  std::vector<node_type> seed_node_types(1, "A");
  pyg::classes::MetapathTracker tracker(edge_types, num_neighbors,
                                        seed_node_types);
  auto b1A = tracker.init_batch(1, "A", 1);
  auto b2A = tracker.init_batch(2, "A", 10);
  EXPECT_EQ(b1A, b2A);
  auto b1AB = tracker.get_neighbor_metapath(b1A, "A__to__B");
  EXPECT_NE(b1AB, b1A);
  EXPECT_EQ(tracker.get_sample_size(1, b1A, {"A", "to", "B"}), 10);
  EXPECT_EQ(tracker.get_sample_size(2, b2A, {"A", "to", "B"}), 100);
  tracker.report_sample_size(1, b1AB, 5);
  tracker.report_sample_size(2, b1AB, 25);
  auto b1ABC = tracker.get_neighbor_metapath(b1AB, "B__to__C");
  EXPECT_NE(b1ABC, b1AB);
  EXPECT_NE(b1ABC, b1A);
  EXPECT_EQ(tracker.get_sample_size(1, b1AB, {"B", "to", "C"}), 20);
  EXPECT_EQ(tracker.get_sample_size(2, b1AB, {"B", "to", "C"}), 200);
}
