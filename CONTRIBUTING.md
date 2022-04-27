## Build & Test

`pyg-lib` can be build as a standalone C++/CUDA library.
For this, we adopt [CMake](https://cmake.org/) to build and [GoogleTest](https://github.com/google/googletest) to test.

First, install all requirements:

```
conda install cmake ninja
```

Then, build the library via:

```
mkdir build
cd build
export Torch_DIR=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
cmake .. -GNinja -DBUILD_TEST=ON -DWITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build .
```

If you want to include the test suite, specify `-DBUILD_TEST=ON` (`OFF` by default).
Finally, run tests via:
```
ctest --verbose --output-on-failure
```

You can also build `pyg-lib` as a Python library via `pip install -e .` from the root directory which uses a `CMakeExtension` to build the C++ library internally.
