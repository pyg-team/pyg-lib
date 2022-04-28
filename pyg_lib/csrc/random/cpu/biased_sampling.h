#pragma once

#include <algorithm>
#include <iterator>

#include "pyg_lib/csrc/random/cpu/rand_engine.h"

namespace pyg {
namespace random {

// The two following functions randomly choose one index according to the given
// bias array.
// The probability for an index to be chosen is proportional to the
// corresponding bias.

/**
 * Biased random choice from a preprocessed CDF (exclusive sum) bias array
 *
 * @tparam Idx type of the set of indices to choose from
 *
 * @tparam Bias type of the weight array
 *
 * @param idx the pointer to the start of the index array
 *
 * @param bias the pointer to the start of the bias array
 *
 * @param len the length of the index/bias arraay
 *
 * @param sum the sum of all bias in CDF (usually 1.0), decided by preprocessing
 *
 * @returns the chosen index
 *
 */

template <typename Idx, typename Bias>
Idx biased_random_cdf(const Idx* idx,
                      const Bias* bias,
                      int len,
                      Bias sum,
                      RandrealEngine<Bias>& eng) {
  Bias rand = eng() * sum;
  auto iter = std::lower_bound(bias, bias + len, rand);
  auto diff = std::distance(bias, iter);
  return idx[diff - 1];
}

template <typename Idx, typename Bias>
Idx biased_random_alias(const Idx* idx,
                        const Idx* alias,
                        const Bias* bias,
                        int len,
                        Bias sum,
                        RandrealEngine<Bias>& eng);
}  // namespace random

}  // namespace pyg
