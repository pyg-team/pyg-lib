include(FetchContent)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark
  GIT_TAG v1.7.0
)
FetchContent_MakeAvailable(googlebenchmark)

set(CBENCHMARK benchmark/csrc)
file(GLOB_RECURSE ALL_BENCHMARKS ${CBENCHMARK}/*.cpp)

foreach(benchmark ${ALL_BENCHMARKS})
    get_filename_component(name ${benchmark} NAME_WE)
    add_executable(${name} ${benchmark})
    target_link_libraries(${name} ${PROJECT_NAME} benchmark::benchmark benchmark::benchmark_main torch)
    target_include_directories(${name} PRIVATE ${PHMAP_DIR})
endforeach()
