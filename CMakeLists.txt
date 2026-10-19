cmake_minimum_required(VERSION 3.20)

# Set the cache default before project() initializes compiler flags, so callers
# can select a baseline CPU with -DCMAKE_CXX_FLAGS_RELEASE=... .
# -ffast-math deliberately omitted for bit-comparable reference reductions.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -g" CACHE STRING "Release C++ compiler flags")
project(mk-infer CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(mkinfer_core
  src/safetensors.cpp
  src/tokenizer.cpp
  src/model.cpp
  src/kv_cache.cpp
  src/sampler.cpp
  src/threadpool.cpp
  src/kernels/matmul.cpp
  src/kernels/quant.cpp
  src/kernels/ops.cpp
)
target_include_directories(mkinfer_core PUBLIC src third_party)

add_executable(mk-infer src/main.cpp)
target_link_libraries(mk-infer mkinfer_core)
install(TARGETS mk-infer RUNTIME DESTINATION bin)

enable_testing()
add_executable(test_kernels tests/test_kernels.cpp)
target_link_libraries(test_kernels mkinfer_core)
add_test(NAME kernels COMMAND test_kernels)

add_executable(test_forward tests/test_forward.cpp)
target_link_libraries(test_forward mkinfer_core)

add_executable(test_regression tests/test_regression.cpp)
target_link_libraries(test_regression mkinfer_core)
foreach(suite tokenizer quantization)
  add_test(NAME ${suite}_regression COMMAND test_regression
    ${suite} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
  set_tests_properties(${suite}_regression PROPERTIES TIMEOUT 10)
endforeach()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME cpu_smoke COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke_cpu.py --binary $<TARGET_FILE:mk-infer>)
  set_tests_properties(cpu_smoke PROPERTIES TIMEOUT 20)
endif()
