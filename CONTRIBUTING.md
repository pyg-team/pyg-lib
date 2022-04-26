## Build & Test

`pyg_lib` could be built as a standalone C++ library. We adopt [CMake](https://cmake.org/) to build and [GoogleTest](https://github.com/google/googletest) to test our C++ code.


The library can be built with `cmake`. If you want to build tests, `-DBUILD_TEST=ON` (`OFF` by default) should be specified, and you can run them with `ctest`. A possible building and testing example could be found [here](https://github.com/pyg-team/pyg-lib/blob/master/.github/workflows/testing.yml).

You can also build `pyg_lib` as a Python library using `pip install .` with our C++ library as a `CMakeExtension`, which will be loaded (`libpyg.so`) when you `import pyg_lib`
