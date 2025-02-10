#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "pyg_lib/csrc/classes/cpu/hash_map.cpp"

TEST(HashMapTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);
  auto key = at::tensor({0, 10, 30, 20}, options);

  auto map = pyg::classes::CPUHashMap(key);
  EXPECT_EQ(map.size(), 4);
  EXPECT_EQ(map.dtype(), at::kLong);
  EXPECT_EQ(map.device(), at::Device(at::kCPU));
  EXPECT_TRUE(at::equal(map.keys(), key));

  auto query = at::tensor({30, 10, 20, 40}, options);
  auto expected = at::tensor({2, 1, 3, -1}, options);
  EXPECT_TRUE(at::equal(map.get(query), expected));

  map = pyg::classes::CPUHashMap(key, 16);
  EXPECT_TRUE(at::equal(map.keys(), key));
  EXPECT_TRUE(at::equal(map.get(query), expected));
}
