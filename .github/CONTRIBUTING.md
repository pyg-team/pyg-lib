## Build & Test

`pyg-lib` can be built as a standalone C++/CUDA library.
For this, we adopt [CMake](https://cmake.org/) to build, [GoogleTest](https://github.com/google/googletest) to test
and [GoogleBenchmark](https://github.com/google/benchmark) to benchmark.

First, install all requirements:

```
conda install cmake ninja
```

Clone a copy of `pyg-lib` from source:

```
git clone --recursive https://github.com/pyg-team/pyg-lib
cd pyg-lib
```

If you already previously cloned `pyg-lib`, update it:

```
git pull
git submodule sync --recursive
git submodule update --init --recursive
```

Then, build the library via:

```
mkdir build
cd build
export Torch_DIR=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
export CUDA_ARCH_LIST=75
cmake .. -GNinja -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON -DWITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH_LIST}
cmake --build .
```

If you want to build a library with MKL BLAS support, you need to have MKL headers present on your machine.
In a typical case, where MKL headers are installed via `conda` or `pip`, run following commands:

```
export BLAS_INCLUDE_DIR=`python -c 'import sysconfig;print(sysconfig.get_path("data"))'`/include
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON -DUSE_MKL_BLAS=ON -DBLAS_INCLUDE_DIR=${BLAS_INCLUDE_DIR}
```

If you want to include the test suite or benchmarks, specify `-DBUILD_TEST=ON` or `-DBUILD_BENCHMARK=ON`, respectively
(both are `OFF` by default).
Finally, run tests via:
```
ctest --verbose --output-on-failure
```

You can also build `pyg-lib` as a Python library via `pip install -e .` which uses a `CMakeExtension` to build the C++ library internally.
