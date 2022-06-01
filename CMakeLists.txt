cmake_minimum_required(VERSION 3.12)
project(pyg)
set(CMAKE_CXX_STANDARD 14)
set(PYG_VERSION 0.1.0)

option(BUILD_TEST "Enable testing" OFF)
option(WITH_COV "Enable code coverage" OFF)
option(USE_PYTHON "Link to Python when building" OFF)
option(WITH_CUDA "Enable CUDA support" OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

if (USE_PYTHON)
  add_definitions(-DUSE_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Development)
endif()

if(WITH_CUDA)
  enable_language(CUDA)
  add_definitions(-DWITH_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

  set(CUTLASS_DIR third_party/cutlass/include)
  include_directories(${CUTLASS_DIR})
  set(CUTLASS_UTIL_DIR third_party/cutlass/tools/util/include)
  include_directories(${CUTLASS_UTIL_DIR})
endif()

set(CSRC pyg_lib/csrc)
file(GLOB_RECURSE ALL_SOURCES ${CSRC}/*.cpp)
if (WITH_CUDA)
  file(GLOB_RECURSE ALL_SOURCES ${ALL_SOURCES} ${CSRC}/*.cu)
endif()
file(GLOB_RECURSE ALL_HEADERS ${CSRC}/*.h)
add_library(${PROJECT_NAME} SHARED ${ALL_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(Torch REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

if (USE_PYTHON)
  target_link_libraries(${PROJECT_NAME} PRIVATE Python3::Python)
endif()

if(BUILD_TEST)
  if(WITH_COV)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 --coverage")
  endif()
  include(cmake/test.cmake)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
  EXPORT_NAME PyG
  INSTALL_RPATH ${TORCH_INSTALL_PREFIX}/lib)
