#include <gtest/gtest.h>

#include <vector>

#include "pyg_lib/csrc/random/cpu/biased_sampling.h"
#include "pyg_lib/csrc/random/cpu/rand_engine.h"

TEST(BiasedSamplingCDFRandomTest, BasicAssertions) {
  pyg::random::RandrealEngine<float> eng;

  // Test if it is roughly biased:
  std::vector<float> cdf = {0.0, 0.2, 0.8};
  std::vector<int> idx = {0, 1, 2};

  std::vector<int> cnt(3, 0);

  int iter = 1000;

  for (int i = 0; i < iter; i++) {
    auto choice = pyg::random::biased_random_cdf<int, float>(
        idx.data(), cdf.data(), 3, 1.0, eng);
    cnt[choice]++;
  }

  EXPECT_LT(cnt[0], 0.3 * iter);
  EXPECT_GT(cnt[0], 0.1 * iter);
  EXPECT_LT(cnt[1], 0.7 * iter);
  EXPECT_GT(cnt[1], 0.5 * iter);
  EXPECT_LT(cnt[2], 0.3 * iter);
  EXPECT_GT(cnt[2], 0.1 * iter);
}
