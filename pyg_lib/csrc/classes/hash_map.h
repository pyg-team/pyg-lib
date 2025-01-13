#pragma once

#include <ATen/ATen.h>
#include "hash_map_impl.h"

namespace pyg {
namespace classes {

struct HashMap : torch::CustomClassHolder {
 public:
  HashMap(const at::Tensor& key);
  at::Tensor get(const at::Tensor& query);

 private:
  std::unique_ptr<HashMapImpl> map_;
};

}  // namespace classes
}  // namespace pyg
