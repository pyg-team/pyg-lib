#include <torch/library.h>

namespace pyg {
namespace classes {

struct CPUHashMap : torch::CustomClassHolder {
  CPUHashMap(){};
};

TORCH_LIBRARY(pyg, m) {
  m.class_<CPUHashMap>("CPUHashMap").def(torch::init());
}

}  // namespace classes
}  // namespace pyg
