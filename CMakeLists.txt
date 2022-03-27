cmake_minimum_required(VERSION 3.12)
project(pyg)
set(CMAKE_CXX_STANDARD 14)
set(PYG_VERSION 0.0.0)

option(WITH_CUDA "Enable CUDA support" OFF)
option(USE_PYTHON "Link to Python when building" ON)

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

set(CSRC pyg_lib/csrc)
file(GLOB ALL_SOURCES ${ALL_SOURCES} ${CSRC}/*.*)
add_library(${PROJECT_NAME} SHARED ${ALL_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})

if (USE_PYTHON)
  target_link_libraries(${PROJECT_NAME} PRIVATE Python3::Python)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
  EXPORT_NAME PyG
  INSTALL_RPATH ${TORCH_INSTALL_PREFIX}/lib)
