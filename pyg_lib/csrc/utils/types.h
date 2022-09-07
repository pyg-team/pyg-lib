#pragma once

#include <string>
#include <tuple>

typedef std::string node_type;
typedef std::string rel_type;
typedef std::tuple<std::string, std::string, std::string> edge_type;

inline rel_type to_rel_type(const edge_type& key) {
  return std::get<0>(key) + "__" + std::get<1>(key) + "__" + std::get<2>(key);
}
