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
 * @param len the length of the index/bias array
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
 * index random choice from a preprocessed alias table with bias
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
  return is_alias ? idx[alias[choice]] : idx[choice];
}

/**
 * Give the CDF representation of a biased CSR.
 *
 * @param rowptr the row pointer of an CSR, needed because we want to group the
 * neighbors of each node.
 *
 * @param bias the edge bias array which indicates the sampling weight for each
 * edge.
 *
 * @returns (optional) the cdf array which is grouped by the neighbors of each
 * node. For each group of neighbors, the weight is exclusively summed to form a
 * cdf array. The sum of each group will be guaranteed be equal to 1.
 *
 * Example:
 *
 * Neighbors of a node has the following bias: {0.5, 2.5, 1.0}
 * The cdf of this group will be: {0.0, 0.125, 0.75}
 *
 */
c10::optional<at::Tensor> biased_to_cdf(const at::Tensor& rowptr,
                                        at::Tensor& bias,
                                        bool inplace);

// The implementation of converting to CDF representation for biased sampling.
template <typename scalar_t>
void biased_to_cdf_helper(int64_t* rowptr_data,
                          size_t rowptr_size,
                          const scalar_t* bias,
                          scalar_t* cdf);

/**
 * Give the alias table of a biased CSR.
 *
 * @param rowptr the row pointer of an CSR, needed because we want to group the
 * neighbors of each node.
 *
 * @param bias the edge bias array which indicates the sampling weight for each
 * edge.
 *
 * @returns A pair of tensors: {bias, alias}
 *
 * The output alias tensor is the alias index of the corresponding entry. For
 * each entry, we can either sample the entry index itself or the alias index.
 *
 * The output bias is grouped by the neighbors of each node. For
 * each group of neighbors, the number (p) means the probability of sampling the
 * index of this alias table entry instead of its alias index (1 - p).
 *
 * Example:
 *
 * Neighbors of a node has the following bias: {0.5, 2, 0.5}
 * The output bias of this group will be: {0.5, 1.0, 0.5}
 * The alias of this group will be: {1, 1, 1}
 * Because index 1 has a high bias number, 0 and 2 will make 1 the alias index
 * in their alias table entries.
 *
 */
std::pair<at::Tensor, at::Tensor> biased_to_alias(at::Tensor rowptr,
                                                  at::Tensor bias);

// The implementation of converting to alias table for biased sampling.
template <typename scalar_t>
void biased_to_alias_helper(int64_t* rowptr_data,
                            size_t rowptr_size,
                            const scalar_t* bias,
                            scalar_t* out_bias,
                            int64_t* alias);

}  // namespace random

}  // namespace pyg
