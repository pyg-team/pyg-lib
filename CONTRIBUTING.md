## Build & Test

`pyg_lib` could be built as a standalone C++ library. We adopt [CMake](https://cmake.org/) to build and [GoogleTest](https://github.com/google/googletest) to test our C++ code.

The library can be built with `cmake`. If you want to build tests, `-DBUILD_TEST=ON` (`OFF` by default) should be specified, and you can run it with `ctest`. A possible building and testing example could be found [here](https://github.com/pyg-team/pyg-lib/blob/master/.github/workflows/install.yml).
