#include <gtest/gtest.h>

#include <vector>

#include "pyg_lib/csrc/random/cpu/biased_sampling.h"
#include "pyg_lib/csrc/random/cpu/rand_engine.h"
#include "pyg_lib/csrc/utils/cpu/convert.h"
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

  std::vector<float> bias_vec{1.5, 0.5, 1.0, 0.25, 0.125, 0.375, 1.0, 1.0};
  std::vector<float> cdf_vec{0.0, 0.75, 0.0, 0.8, 0.0, 0.25, 0.0, 0.5};

  at::Tensor bias = pyg::utils::from_vector<float>(bias_vec);
  at::Tensor cdf = pyg::utils::from_vector<float>(cdf_vec);

  auto res = pyg::random::biased_to_cdf(rowptr, bias, false);

  EXPECT_TRUE(at::equal(res.value(), cdf));

  pyg::random::biased_to_cdf(rowptr, bias, true);

  EXPECT_TRUE(at::equal(bias, cdf));
}

TEST(BiasedSamplingAliasConversionTest, BasicAssertions) {
  auto options = at::TensorOptions().dtype(at::kLong);

  auto graph = cycle_graph(/*num_nodes=*/4, options);
  auto rowptr = std::get<0>(graph);

  std::vector<float> bias_vec{1.5, 0.5, 0.75, 0.25, 0.125, 0.375, 1.0, 1.0};
  std::vector<float> out_vec{1.0, 0.5, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0};
  std::vector<int64_t> alias_vec{0, 0, 0, 0, 1, 1, 0, 1};

  at::Tensor bias = pyg::utils::from_vector<float>(bias_vec);
  at::Tensor out_bias = pyg::utils::from_vector<float>(out_vec);
  at::Tensor alias = pyg::utils::from_vector<int64_t>(alias_vec);

  auto res = pyg::random::biased_to_alias(rowptr, bias);
  auto res_bias = std::get<0>(res);
  auto res_alias = std::get<1>(res);

  EXPECT_TRUE(at::equal(res_bias, out_bias));

  EXPECT_TRUE(at::equal(res_alias, alias));

  // Test with a longer neighborhood array
  std::vector<int64_t> long_rowptr_vec{0, 4};
  std::vector<float> long_bias_vec{0.75, 0.5, 1.5, 1.25};
  std::vector<float> long_out_vec{0.75, 0.5, 1.0, 0.75};
  std::vector<int64_t> long_alias_vec{2, 3, 2, 2};

  at::Tensor long_rowptr = pyg::utils::from_vector<int64_t>(long_rowptr_vec);
  at::Tensor long_bias = pyg::utils::from_vector<float>(long_bias_vec);
  at::Tensor long_out_bias = pyg::utils::from_vector<float>(long_out_vec);
  at::Tensor long_alias = pyg::utils::from_vector<int64_t>(long_alias_vec);

  auto long_res = pyg::random::biased_to_alias(long_rowptr, long_bias);
  auto long_res_bias = std::get<0>(long_res);
  auto long_res_alias = std::get<1>(long_res);

  EXPECT_TRUE(at::equal(long_res_bias, long_out_bias));

  EXPECT_TRUE(at::equal(long_res_alias, long_alias));
}
