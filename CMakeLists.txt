cmake_minimum_required(VERSION 3.20)
project(mk-infer CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -ffast-math deliberately omitted for now: Phase 2 requires bit-comparable
# reductions against the reference dumps. Revisit in Phase 4 with measurements.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -g")

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

enable_testing()
add_executable(test_kernels tests/test_kernels.cpp)
target_link_libraries(test_kernels mkinfer_core)
add_test(NAME kernels COMMAND test_kernels)

add_executable(test_forward tests/test_forward.cpp)
target_link_libraries(test_forward mkinfer_core)
