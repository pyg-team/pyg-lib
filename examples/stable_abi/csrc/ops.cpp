#include <Python.h>

#include <torch/csrc/stable/library.h>

namespace pyg {

int64_t cuda_version() {
#ifdef WITH_CUDA
  return CUDA_VERSION;
#else
  return -1;
#endif
}

STABLE_TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def("cuda_version() -> int");
}

STABLE_TORCH_LIBRARY_IMPL(pyg, CompositeExplicitAutograd, m) {
  m.impl("cuda_version", TORCH_BOX(&cuda_version));
}

}  // namespace pyg

extern "C" {
PyObject* PyInit__C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT, "_C", NULL, -1, NULL,
  };
  return PyModule_Create(&module_def);
}
}
