cmake_minimum_required(VERSION 3.12)
project(pyg)
set(CMAKE_CXX_STANDARD 14)
set(PYG_VERSION 0.0.0)

option(WITH_CUDA "Enable CUDA support" OFF)
option(USE_PYTHON "Link to Python when building" OFF)

if(WITH_CUDA)
  enable_language(CUDA)
  add_definitions(-DWITH_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

if (USE_PYTHON)
  add_definitions(-DUSE_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Development)
endif()

find_package(Torch REQUIRED)
