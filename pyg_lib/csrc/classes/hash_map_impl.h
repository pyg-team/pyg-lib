#pragma once

#include <ATen/ATen.h>

namespace pyg {
namespace classes {

struct HashMapImpl {
  virtual ~HashMapImpl() = default;
  virtual at::Tensor get(const at::Tensor& query) = 0;
};

}  // namespace classes
}  // namespace pyg
