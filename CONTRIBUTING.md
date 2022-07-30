## Build & Test

`pyg-lib` can be built as a standalone C++/CUDA library.
For this, we adopt [CMake](https://cmake.org/) to build and [GoogleTest](https://github.com/google/googletest) to test.

First, install all requirements:

```
conda install cmake ninja
```

Clone a copy of `pyg-lib` from source:

```
git clone --recursive https://github.com/pyg-team/pyg-lib
cd pg-lib
```

If you already previously cloned `pyg-lib`, update it:

```
git pull
git submodule sync --recursive
git submodule update --init --recursive --jobs 0
```

Then, build the library via:

```
mkdir build
cd build
export Torch_DIR=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
export CUDA_ARCH_LIST=75
cmake .. -GNinja -DBUILD_TEST=ON -DWITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH_LIST}
cmake --build .
```

If you want to include the test suite, specify `-DBUILD_TEST=ON` (`OFF` by default).
Finally, run tests via:
```
ctest --verbose --output-on-failure
```

You can also build `pyg-lib` as a Python library via `pip install -e .` which uses a `CMakeExtension` to build the C++ library internally.
