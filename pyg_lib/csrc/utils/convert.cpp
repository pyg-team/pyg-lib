#include "convert.h"

namespace pyg {
namespace utils {

at::Tensor size_from_ptr(const at::Tensor& ptr) {
  return ptr.narrow(/*dim=*/0, /*start=*/1, /*length=*/ptr.numel() - 1) -
         ptr.narrow(/*dim=*/0, /*start=*/0, /*length=*/ptr.numel() - 1);
}

}  // namespace utils
}  // namespace pyg
