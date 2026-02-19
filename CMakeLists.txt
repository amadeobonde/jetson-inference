cmake_minimum_required(VERSION 3.18)
project(jetson-inference LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 87)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# --- Options ---
option(JINF_BUILD_TESTS "Build unit tests" ON)
option(JINF_BUILD_BENCH "Build benchmarks" ON)
option(JINF_BUILD_TOOLS "Build CLI tools" ON)

# --- Dependencies ---
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(Liburing REQUIRED)

include(JetsonArch)

# --- Libraries ---
add_subdirectory(src)
add_subdirectory(cuda)

# --- Optional targets ---
if(JINF_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(JINF_BUILD_BENCH)
    add_subdirectory(bench)
endif()

if(JINF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
