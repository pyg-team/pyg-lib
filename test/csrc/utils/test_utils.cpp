#include <gtest/gtest.h>

#include "pyg_lib/csrc/sampler/subgraph.h"

TEST(UtilsTypeTest, BasicAssertions) {
  pyg::utils::edge_t edge = "node1__to__node2";

  auto src = pyg::utils::get_src(edge);
  auto dst = pyg::utils::get_dst(edge);
  auto rel = pyg::utils::get_rel(edge);

  EXPECT_EQ(src, std::string("node1"));
  EXPECT_EQ(dst, std::string("node2"));
  EXPECT_EQ(rel, std::string("to"));
}
