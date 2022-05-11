#pragma once

#include <string>

#include <ATen/ATen.h>

namespace pyg {
namespace utils {

const std::string SPLIT_TOKEN = "__";

using edge_t = std::string;
using node_t = std::string;
using rel_t = std::string;

using edge_tensor_dict_t = c10::Dict<edge_t, at::Tensor>;
using node_tensor_dict_t = c10::Dict<node_t, at::Tensor>;

inline node_t get_src(const edge_t& e) {
  return e.substr(0, e.find_first_of(SPLIT_TOKEN));
}

inline rel_t get_rel(const edge_t& e) {
  auto beg = e.find_first_of(SPLIT_TOKEN) + SPLIT_TOKEN.size();
  return e.substr(beg,
                  e.find_last_of(SPLIT_TOKEN) - SPLIT_TOKEN.size() + 1 - beg);
}

inline node_t get_dst(const edge_t& e) {
  return e.substr(e.find_last_of(SPLIT_TOKEN) + 1);
}
}  // namespace utils

}  // namespace pyg
