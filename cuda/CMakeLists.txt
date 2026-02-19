# cuda/CMakeLists.txt — CUDA kernels library

add_library(jinf_cuda STATIC
    dequant_kernels.cu
    matmul_kernels.cu
    rmsnorm_kernel.cu
    rope_kernel.cu
    silu_kernel.cu
    softmax_kernel.cu
    attention_kernel.cu
    sparse_matmul.cu
)

target_include_directories(jinf_cuda PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(jinf_cuda PUBLIC
    CUDA::cudart
)

set_target_properties(jinf_cuda PROPERTIES
    CUDA_ARCHITECTURES 87
    POSITION_INDEPENDENT_CODE ON
)
