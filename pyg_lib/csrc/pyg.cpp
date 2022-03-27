#include "pyg.h"

#ifdef USE_PYTHON
#include <Python.h>
#endif

#ifdef WITH_CUDA
#include <cuda.h>
#endif

#include <torch/library.h>

// If we are in a Windows environment, we need to define
// initialization functions for the _custom_ops extension.
// For PyMODINIT_FUNC to work, we need to include Python.h
#ifdef _WIN32
#ifdef USE_PYTHON
PyMODINIT_FUNC PyInit__C(void) { return NULL; }
#endif // USE_PYTHON
#endif // _WIN32

namespace pyg {

int64_t cuda_version() {
#ifdef WITH_CUDA
  return CUDA_VERSION;
#else
  return -1;
#endif
}

TORCH_LIBRARY_FRAGMENT(pyg, m) { m.def("cuda_version", &cuda_version); }

} // namespace pyg
