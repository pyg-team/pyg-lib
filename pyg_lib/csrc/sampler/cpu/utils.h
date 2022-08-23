#pragma once

#include <torch/torch.h>

#include "parallel_hashmap/phmap.h"

template <typename scalar_t>
inline torch::Tensor from_vector(const std::vector<scalar_t> &vec,
                                 bool inplace = false) {
  const auto size = (int64_t)vec.size();
  const auto out = torch::from_blob((scalar_t *)vec.data(), {size},
                                    c10::CppTypeToScalarType<scalar_t>::value);
  return inplace ? out : out.clone();
}

template <typename key_t, typename scalar_t>
inline c10::Dict<key_t, torch::Tensor>
from_vector(const phmap::flat_hash_map<key_t, std::vector<scalar_t>> &vec_dict,
            bool inplace = false) {
  c10::Dict<key_t, torch::Tensor> out_dict;
  for (const auto &kv : vec_dict)
    out_dict.insert(kv.first, from_vector<scalar_t>(kv.second, inplace));
  return out_dict;
}

inline int64_t uniform_randint(int64_t low, int64_t high) {
  CHECK_LT(low, high);
  auto options = torch::TensorOptions().dtype(torch::kInt64);
  auto ret = torch::randint(low, high, {1}, options);
  auto ptr = ret.data_ptr<int64_t>();
  return *ptr;
}

inline int64_t uniform_randint(int64_t high) {
  return uniform_randint(0, high);
}
