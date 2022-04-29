#include <gtest/gtest.h>

#include <vector>

#include "pyg_lib/csrc/random/cpu/biased_sampling.h"
#include "pyg_lib/csrc/random/cpu/rand_engine.h"
#include "test/csrc/graph.h"

TEST(BiasedSamplingCDFRandomTest, BasicAssertions) {
  pyg::random::RandrealEngine<float> eng;

  // Test if it is roughly biased:

  // The pdf (bias) array is {0.2, 0.6, 0.2}
  // so the cdf array (exclusive sum) is {0,0, 0.2, 0.8}

  std::vector<float> cdf = {0.0, 0.2, 0.8};
  std::vector<int> idx = {0, 1, 2};

  std::vector<int> cnt(3, 0);

  int iter = 1000;

  for (int i = 0; i < iter; i++) {
    auto choice = pyg::random::biased_random_cdf<int, float>(
        idx.data(), cdf.data(), 3, eng);
    cnt[choice]++;
  }

  // Count distribution should be within [-0.1, +0.1] compared with
  // expectations.

  EXPECT_LT(cnt[0], 0.3 * iter);
  EXPECT_GT(cnt[0], 0.1 * iter);
  EXPECT_LT(cnt[1], 0.7 * iter);
  EXPECT_GT(cnt[1], 0.5 * iter);
  EXPECT_LT(cnt[2], 0.3 * iter);
  EXPECT_GT(cnt[2], 0.1 * iter);
}

TEST(BiasedSamplingAliasRandomTest, BasicAssertions) {
  pyg::random::RandrealEngine<float> eng;

  // Test if it is roughly biased:

  /**
   *     0        1        2
   *  __ __ __ __ __ __ __ __ __
   * |        |        |        |
   * |  0:0.5 |  1:1.0 |  2:0.5 | idx:bias
   * |__ __ __|__ __ __|__ __ __|
   * |        |        |        |
   * |  1:0.5 |  1:0.0 |  1:0.5 | alias:(1 - bias)
   * |__ __ __|__ __ __|__ __ __|
   */

  std::vector<float> bias = {0.5, 1.0, 0.5};
  std::vector<int> alias = {1, 1, 1};
  std::vector<int> idx = {0, 1, 2};

  std::vector<int> cnt(3, 0);

  int iter = 1000;

  for (int i = 0; i < iter; i++) {
    auto choice = pyg::random::biased_random_alias<int, float>(
        idx.data(), alias.data(), bias.data(), 3, eng);
    cnt[choice]++;
  }

  EXPECT_LT(cnt[0], 0.3 * iter);
  EXPECT_GT(cnt[0], 0.1 * iter);
  EXPECT_LT(cnt[1], 0.7 * iter);
  EXPECT_GT(cnt[1], 0.5 * iter);
  EXPECT_LT(cnt[2], 0.3 * iter);
  EXPECT_GT(cnt[2], 0.1 * iter);
}

TEST(BiasedSamplingCDFConversionTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  auto graph = cycle_graph(/*num_nodes=*/4, options);
  auto rowptr = std::get<0>(graph);

  std::vector<float> bias_vec{1.5, 0.5, 0.8, 0.2, 0.1, 0.3, 1.0, 1.0};
  std::vector<float> cdf_vec{0.0, 0.75, 0.0, 0.8, 0.0, 0.25, 0.0, 0.5};

  at::Tensor bias = at::from_blob(bias_vec.data(), {bias_vec.size()},
                                  at::TensorOptions().dtype(at::kFloat));
  at::Tensor cdf = at::from_blob(cdf_vec.data(), {cdf_vec.size()},
                                 at::TensorOptions().dtype(at::kFloat));

  auto res = pyg::random::biased_to_cdf(rowptr, bias);

  EXPECT_TRUE(at::equal(res, cdf));

  pyg::random::biased_to_cdf_inplace(rowptr, bias);

  EXPECT_TRUE(at::equal(bias, cdf));
}
