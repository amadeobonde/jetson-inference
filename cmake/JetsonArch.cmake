# JetsonArch.cmake
# Detect aarch64 / Jetson platform and set appropriate compiler flags.

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(STATUS "Detected aarch64 — enabling ARMv8.2-A flags")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16+dotprod")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -march=armv8.2-a+fp16+dotprod")

    # Check for Jetson / Tegra platform
    if(EXISTS "/etc/nv_tegra_release")
        file(READ "/etc/nv_tegra_release" _NV_TEGRA_CONTENT)
        message(STATUS "Jetson platform detected: ${_NV_TEGRA_CONTENT}")
        set(JINF_IS_JETSON TRUE CACHE BOOL "Running on NVIDIA Jetson platform")
    else()
        set(JINF_IS_JETSON FALSE CACHE BOOL "Running on NVIDIA Jetson platform")
    endif()
else()
    message(STATUS "Non-aarch64 platform (${CMAKE_SYSTEM_PROCESSOR})")
    set(JINF_IS_JETSON FALSE CACHE BOOL "Running on NVIDIA Jetson platform")
endif()
