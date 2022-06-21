#pragma once

#include <string>

#include <ATen/ATen.h>

namespace pyg {
namespace utils {

const std::string SPLIT_TOKEN = "__";

using EdgeType = std::string;
using NodeType = std::string;
using RelationType = std::string;

using EdgeTensorDict = c10::Dict<EdgeType, at::Tensor>;
using NodeTensorDict = c10::Dict<NodeType, at::Tensor>;

inline NodeType get_src(const EdgeType& e) {
  return e.substr(0, e.find_first_of(SPLIT_TOKEN));
}

inline RelationType get_rel(const EdgeType& e) {
  auto beg = e.find_first_of(SPLIT_TOKEN) + SPLIT_TOKEN.size();
  return e.substr(beg,
                  e.find_last_of(SPLIT_TOKEN) - SPLIT_TOKEN.size() + 1 - beg);
}

inline NodeType get_dst(const EdgeType& e) {
  return e.substr(e.find_last_of(SPLIT_TOKEN) + 1);
}
}  // namespace utils

}  // namespace pyg
