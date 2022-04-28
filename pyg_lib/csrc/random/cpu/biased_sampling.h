#pragma once

namespace pyg {
namespace random {

// TODO: Biased random sampling methods

template <typename Idx, typename Bias>
Idx biased_random_cdf(const Idx* idx,
                      const Bias* bias,
                      int len,
                      Bias sum = 1.0);
template <typename Idx, typename Bias>
Idx biased_random_alias(const Idx* idx,
                        const Idx* alias,
                        const Bias* bias,
                        int len,
                        Bias sum = 1.0);
}  // namespace random

}  // namespace pyg
