cmake_minimum_required(VERSION 3.18)
project(pyg)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_SHARED_LIBRARY_PREFIX "lib")
set(PYG_VERSION 0.4.0)
set(NO_METIS 0)

option(BUILD_TEST "Enable testing" OFF)
option(BUILD_BENCHMARK "Enable benchmarks" OFF)
option(WITH_COV "Enable code coverage" OFF)
option(WITH_CUDA "Enable CUDA support" OFF)

if(NOT WIN32 AND NOT DEFINED USE_CXX11_ABI)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  execute_process(
      COMMAND ${Python3_EXECUTABLE} "-c"
              "import torch; print(torch.compiled_with_cxx11_abi(),end='');"
      RESULT_VARIABLE _PYTHON_SUCCESS
      OUTPUT_VARIABLE USE_CXX11_ABI)
  # Convert the bool variable to integer.
  if(USE_CXX11_ABI)
    set(USE_CXX11_ABI 1)
  else()
    set(USE_CXX11_ABI 0)
  endif()
  message(STATUS "Using USE_CXX11_ABI set by PyTorch to ${USE_CXX11_ABI}")
else()
  set(USE_CXX11_ABI 0)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI}")

set(WITH_MKL_BLAS 0)
if(USE_MKL_BLAS AND DEFINED BLAS_INCLUDE_DIR)
  find_file(MKL_INCLUDE_FOUND mkl.h ${BLAS_INCLUDE_DIR} NO_DEFAULT_PATH)
  if(MKL_INCLUDE_FOUND)
    set(WITH_MKL_BLAS 1)
  else()
    if(WITH_COV)
      message(FATAL_ERROR "The mkl.h file was not found - pass the correct directory or set USE_MKL_BLAS=OFF.")
    else()
      message(WARNING "The mkl.h file was not found - building pyg-lib without MKL BLAS support.")
    endif()
  endif()
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pyg_lib/csrc/config.h.in "${CMAKE_CURRENT_SOURCE_DIR}/pyg_lib/csrc/config.h")

if(WITH_CUDA)
  enable_language(CUDA)
  add_definitions(-DWITH_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -allow-unsupported-compiler")

  if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;90;100;120")
  elseif (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
    set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;90")
  elseif (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.3)
    set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")
  else()
    set(CMAKE_CUDA_ARCHITECTURES "60;70;75")
  endif()

  if (NOT "$ENV{EXTERNAL_CUTLASS_INCLUDE_DIR}" STREQUAL "")
    include_directories($ENV{EXTERNAL_CUTLASS_INCLUDE_DIR})
  else()
    set(CUTLASS_DIR third_party/cutlass/include)
    include_directories(${CUTLASS_DIR})
    set(CUTLASS_UTIL_DIR third_party/cutlass/tools/util/include)
    include_directories(${CUTLASS_UTIL_DIR})
  endif()
  set(CUCOLLECTIONS_DIR third_party/cuCollections/include)
  include_directories(${CUCOLLECTIONS_DIR})
endif()

set(CSRC pyg_lib/csrc)
file(GLOB_RECURSE ALL_SOURCES ${CSRC}/*.cpp)
if (WITH_CUDA)
  file(GLOB_RECURSE ALL_SOURCES ${ALL_SOURCES} ${CSRC}/*.cu)
endif()
add_library(${PROJECT_NAME} SHARED ${ALL_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
if(MKL_INCLUDE_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${BLAS_INCLUDE_DIR})
endif()

if (NOT "$ENV{EXTERNAL_PHMAP_INCLUDE_DIR}" STREQUAL "")
  include_directories($ENV{EXTERNAL_PHMAP_INCLUDE_DIR})
else()
  set(PHMAP_DIR third_party/parallel-hashmap)
  include_directories(${PHMAP_DIR})
  target_include_directories(${PROJECT_NAME} PRIVATE ${PHMAP_DIR})
endif()

if (MSVC)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /DIDXTYPEWIDTH=64 /DREALTYPEWIDTH=32")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DIDXTYPEWIDTH=64 /DREALTYPEWIDTH=32")
else()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DIDXTYPEWIDTH=64 -DREALTYPEWIDTH=32")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIDXTYPEWIDTH=64 -DREALTYPEWIDTH=32")
endif()

if (NOT MSVC)
  set(METIS_DIR third_party/METIS)
  target_include_directories(${PROJECT_NAME} PRIVATE ${METIS_DIR}/include)
  set(GKLIB_PATH "${METIS_DIR}/GKlib")
  include(${GKLIB_PATH}/GKlibSystem.cmake)
  include_directories(${GKLIB_PATH})
  include_directories("${METIS_DIR}/include")
  add_subdirectory("${METIS_DIR}/libmetis")
  target_link_libraries(${PROJECT_NAME} PRIVATE metis)
endif()

find_package(Torch REQUIRED)
message("-- TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

if(WITH_CUDA)
  target_include_directories(${PROJECT_NAME} PRIVATE
                             third_party/cccl/thrust
                             third_party/cccl/cub
                             third_party/cccl/libcudacxx/include)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

if(BUILD_TEST)
  if(WITH_COV)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} --coverage")
  endif()
  include(cmake/test.cmake)
endif()

message("-- CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message("-- CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message("-- CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")

if(BUILD_BENCHMARK)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE INTERNAL "Disable benchmarking tool tests")
  set(COMPILE_HAVE_GNU_POSIX_REGEX OFF CACHE INTERNAL "Disable GNU POSIX regex compilation check")
  include(cmake/benchmark.cmake)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
  EXPORT_NAME PyG
  INSTALL_RPATH ${TORCH_INSTALL_PREFIX}/lib)

# Cmake creates *.dylib by default, but python expects *.so by default
if (APPLE)
  set_property(TARGET ${PROJECT_NAME} PROPERTY SUFFIX .so)
elseif (MSVC)
  set_property(TARGET ${PROJECT_NAME} PROPERTY SUFFIX .pyd)
endif()
