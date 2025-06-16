#include "library.h"

#ifdef WITH_CUDA
#include <cuda.h>
#endif

#include <torch/library.h>

// If we are in a Windows environment, we need to define
// initialization functions for the _custom_ops extension.
#ifdef _WIN32
void* PyInit__C(void) {
  return NULL;
}
#endif  // _WIN32

namespace pyg {

int64_t cuda_version() {
#ifdef WITH_CUDA
  return CUDA_VERSION;
#else
  return -1;
#endif
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def("cuda_version", &cuda_version);
}

}  // namespace pyg
