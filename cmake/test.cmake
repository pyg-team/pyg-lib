include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

set(CTEST test/csrc)
file(GLOB_RECURSE ALL_TESTS ${CTEST}/*.cpp)

foreach(t ${ALL_TESTS})
    get_filename_component(name ${t} NAME_WE)
    add_executable(${name} ${t})
    target_link_libraries(${name} ${PROJECT_NAME} gtest_main)
    target_include_directories(${name} PRIVATE ${CSRC}/include)
    gtest_discover_tests(${name})
endforeach()
