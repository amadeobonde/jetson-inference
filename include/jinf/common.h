#pragma once

#include <cstdint>
#include <cstddef>
#include <chrono>

// ---- Status codes ----

enum jinf_status : int {
    JINF_OK          = 0,
    JINF_ERR_IO      = 1,
    JINF_ERR_PARSE   = 2,
    JINF_ERR_CUDA    = 3,
    JINF_ERR_OOM     = 4,
    JINF_ERR_INVALID = 5,
};

inline const char* jinf_status_str(jinf_status s) {
    switch (s) {
        case JINF_OK:          return "OK";
        case JINF_ERR_IO:      return "I/O error";
        case JINF_ERR_PARSE:   return "parse error";
        case JINF_ERR_CUDA:    return "CUDA error";
        case JINF_ERR_OOM:     return "out of memory";
        case JINF_ERR_INVALID: return "invalid argument";
        default:               return "unknown error";
    }
}

// ---- Alignment macros ----

#define JINF_ALIGN_UP(x, align)  (((x) + ((align) - 1)) & ~((align) - 1))
#define JINF_ALIGN_4K(x)         JINF_ALIGN_UP((x), 4096)

// ---- Timer ----

struct jinf_timer {
    using clock = std::chrono::high_resolution_clock;

    clock::time_point start;

    jinf_timer() { reset(); }

    void reset() { start = clock::now(); }

    double elapsed_ms() const {
        auto now = clock::now();
        return std::chrono::duration<double, std::milli>(now - start).count();
    }

    double elapsed_us() const {
        auto now = clock::now();
        return std::chrono::duration<double, std::micro>(now - start).count();
    }
};

// ---- Logging ----

#include <cstdio>

#define JINF_LOG(fmt, ...) \
    fprintf(stderr, "[jinf] " fmt "\n", ##__VA_ARGS__)

#define JINF_CHECK(status, msg) \
    do { \
        jinf_status _s = (status); \
        if (_s != JINF_OK) { \
            JINF_LOG("ERROR %s: %s (at %s:%d)", jinf_status_str(_s), (msg), __FILE__, __LINE__); \
            return _s; \
        } \
    } while (0)

#define JINF_CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) { \
            JINF_LOG("CUDA error: %s (at %s:%d)", cudaGetErrorString(_err), __FILE__, __LINE__); \
            return JINF_ERR_CUDA; \
        } \
    } while (0)
