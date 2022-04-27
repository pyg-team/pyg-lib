#include <gtest/gtest.h>

#include <set>
#include <vector>

#include "../../../pyg_lib/csrc/random/cpu/randint_engine.h"

TEST(RandintRandomTest, BasicAssertions) {
  pyg::random::RandintEngine<int64_t> eng;

  // Test if it is roughly random
  std::set<int> picked;

  int iter = 1000;
  int beg = 11111111;
  int end = 99999999;

  for (int i = 0; i < iter; i++) {
    auto res = eng(beg, end);
    picked.insert(res);
  }
  EXPECT_EQ(iter, picked.size());
}

TEST(RandintPrefetchTest, BasicAssertions) {
  pyg::random::RandintEngine<int64_t> eng;

  // Test many times to enable prefetching

  int iter = 10000;
  int64_t beg = 86421357;
  int64_t end = 97538642;

  for (int i = 0; i < iter; i++) {
    auto res = eng(beg, end);
    EXPECT_LT(res, end);
    EXPECT_GE(res, beg);
  }
}

TEST(RandintValidTest, BasicAssertions) {
  pyg::random::RandintEngine<int64_t> eng;

  // Test ranges

  std::vector<unsigned> test_bits{10, 20, 30, 40, 50};

  for (auto b : test_bits) {
    int64_t beg = 321;
    int64_t end = beg + (1ULL << b);
    auto res = eng(beg, end);
    EXPECT_LT(res, end);
    EXPECT_GE(res, beg);
  }

  // Test types

  pyg::random::RandintEngine<unsigned short> eng_short_unsigned;
  int64_t beg = 12345;
  int64_t end = 54321;
  auto res_short_unsigned = eng(beg, end);
  EXPECT_LT(res_short_unsigned, end);
  EXPECT_GE(res_short_unsigned, beg);

  pyg::random::RandintEngine<unsigned> eng_int_unsigned;
  beg = 12345678;
  end = 87654321;
  auto res_int_unsigned = eng(beg, end);
  EXPECT_LT(res_int_unsigned, end);
  EXPECT_GE(res_int_unsigned, beg);

  pyg::random::RandintEngine<int> eng_int_signed;
  beg = 12345678;
  end = 87654321;
  auto res_int_signed = eng(beg, end);
  EXPECT_LT(res_int_signed, end);
  EXPECT_GE(res_int_signed, beg);
}

TEST(RandintSeedTest, BasicAssertions) {
  int64_t beg = 12345678;
  int64_t end = 87654321;

  at::manual_seed(147);
  pyg::random::RandintEngine<int64_t> eng1;

  std::vector<int64_t> res;
  for (int i = 0; i < 100; i++) {
    res.push_back(eng1(beg, end));
  }

  at::manual_seed(147);
  pyg::random::RandintEngine<int64_t> eng2;
  for (auto r : res) {
    EXPECT_EQ(eng2(beg, end), r);
  }
}
