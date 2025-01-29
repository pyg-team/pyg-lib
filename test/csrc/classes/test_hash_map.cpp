#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <limits>

#include "pyg_lib/csrc/classes/hash_map.h"

TEST(HashMapTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);
  auto key = at::tensor({0, 10, 30, 20}, options);

  auto map = pyg::classes::HashMap(key);

  auto query = at::tensor({30, 10, 20, 40}, options);
  auto expected = at::tensor({2, 1, 3, -1}, options);
  EXPECT_TRUE(at::equal(map.get(query), expected));
}
