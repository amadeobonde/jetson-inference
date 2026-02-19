# bench/CMakeLists.txt — Benchmarks

add_executable(nvme_bench nvme_bench.cpp)
target_link_libraries(nvme_bench PRIVATE jinf_core)

add_executable(inference_bench inference_bench.cpp)
target_link_libraries(inference_bench PRIVATE jinf_core)
