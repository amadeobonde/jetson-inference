# tests/CMakeLists.txt — Unit tests

# Simple test framework: each test is a standalone executable
# that returns 0 on success, non-zero on failure.

function(jinf_add_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE jinf_core)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

jinf_add_test(test_gguf test_gguf.cpp)
jinf_add_test(test_nvme_io test_nvme_io.cpp)
jinf_add_test(test_splitter test_splitter.cpp)
jinf_add_test(test_buffer_pool test_buffer_pool.cpp)
jinf_add_test(test_engine test_engine.cpp)
jinf_add_test(test_sparse_matmul test_sparse_matmul.cpp)
jinf_add_test(test_activation_predictor test_activation_predictor.cpp)
