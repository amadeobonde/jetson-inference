# src/CMakeLists.txt — Core C++ library

add_library(jinf_core STATIC
    common.cpp
    quant.cpp
    gguf.cpp
    nvme_io.cpp
    nvmw.cpp
    splitter.cpp
    buffer_pool.cpp
    engine.cpp
    activation_predictor.cpp
)

target_include_directories(jinf_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(jinf_core PUBLIC
    Liburing::Liburing
    Threads::Threads
    CUDA::cudart
    jinf_cuda
)

set_target_properties(jinf_core PROPERTIES
    CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
)
