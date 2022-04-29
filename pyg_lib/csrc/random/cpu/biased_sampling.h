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
 * scalar_ted random choice from a preprocessed CDF (exclusive sum) bias array
 *
 * @tparam index_t type of the set of indices to choose from
 *
 * @tparam scalar_t type of the weight array
 *
 * @param idx the pointer to the start of the index array
 *
 * @param bias the pointer to the start of the bias array
 *
 * @param len the length of the index/bias arraay
 *
 * @returns the chosen index
 *
 * Use binary search to map a 0-1 uniform random real number X to idx[i]
 * s.t. idx[i] < X < idx[i+1]
 *
 * Example:
 * idx: { 0,   1,  |  2 }
 * cdf: {0.0, 0.2, | 0.8}
 *                 |
 *              X:0.7
 * -> choose idx = 1
 */

template <typename index_t, typename scalar_t>
index_t biased_random_cdf(const index_t* idx,
                          const scalar_t* cdf,
                          int len,
                          RandrealEngine<scalar_t>& eng) {
  scalar_t rand = eng();
  auto iter = std::lower_bound(cdf, cdf + len, rand);
  auto diff = std::distance(cdf, iter);
  return idx[diff - 1];
}

/**
 * scalar_ted random choice from a preprocessed alias table with bias
 *
 * Reference:
 *
 * [An Efficient Method for Generating Discrete Random Variables with General
 * Distributions](https://dl.acm.org/doi/pdf/10.1145/355744.355749)
 *
 * @tparam index_t type of the set of indices to choose from
 *
 * @tparam scalar_t type of the weight array
 *
 * @param idx the pointer to the start of the index array
 *
 * @param alias the pointer to the start of the alias table
 *
 * @param bias the pointer to the start of the bias array
 *
 * @param len the length of the index array
 *
 * @returns the chosen index
 *
 * 1. Sample uniformly to pick an entry index in the alias table
 * 2. Bionomial sample to choose between the index (bias) and alias (1-bias)
 *
 * Example:
 *
 *     0        1        2
 *  __ __ __ __ __ __ __ __ __
 * |        |        |        |
 * |  0:0.5 |  1:1.0 |  2:0.5 | idx:bias
 * |__ __ __|__ __ __|__ __ __|
 * |        |        |        |
 * |  1:0.5 |  1:0.0 |  1:0.5 | alias:(1-bias)
 * |__ __ __|__ __ __|__ __ __|
 *
 *    The original alias table
 *
 *
 *     0        1        2
 *  __ __ __ __ __ __ __ __ __
 * |        |        |        |
 * |  0:0.5 |  1:1.0 |  2:0.5 | idx:bias
 * |__ __ __|__ __ __|__ __ __|
 * |        |        |        |
 * |  1:0.5 |  1:0.0 |  1:0.5 | alias:(1-bias)
 * |__ __ __|__ __ __|__ __ __|
 *     /|\
 *      |
 *      |
 * (1) Uniformly sample entry 0 from all entries
 *
 *
 *     0        1        2
 *  __ __ __ __ __ __ __ __ __
 * |        |        |        |
 * |   X    |  1:1.0 |  2:0.5 | idx:bias
 * |__ __ __|__ __ __|__ __ __|
 * |        |        |        |
 * |   O    |  1:0.0 |  1:0.5 | alias:(1 - bias)
 * |__ __ __|__ __ __|__ __ __|
 *     /|\
 *      |
 *      |
 * (2) Sample (with bias) alias instead of idx
 *
 */

template <typename index_t, typename scalar_t>
index_t biased_random_alias(const index_t* idx,
                            const index_t* alias,
                            const scalar_t* bias,
                            int len,
                            RandrealEngine<scalar_t>& eng) {
  scalar_t rand = eng();
  int choice = rand * len;
  bool is_alias = eng() > bias[choice];
  return is_alias ? alias[choice] : idx[choice];
}
}  // namespace random

}  // namespace pyg
