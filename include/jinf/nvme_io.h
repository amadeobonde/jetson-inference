#pragma once

#include "jinf/common.h"
#include <cstddef>
#include <cstdint>

// ---- Configuration ----

struct jinf_io_config {
    int    queue_depth;   // io_uring queue depth (default 64)
    size_t read_size;     // preferred read chunk size (default 1MB)
    int    flags;         // reserved
};

static inline jinf_io_config jinf_io_config_default() {
    return { .queue_depth = 64, .read_size = 1024 * 1024, .flags = 0 };
}

// ---- Request / Completion ----

struct jinf_io_request {
    int    fd;
    void*  buffer;        // must be 4K-aligned for O_DIRECT
    size_t offset;        // file offset (must be 4K-aligned)
    size_t length;        // read length (must be 4K-aligned)
    void*  user_data;     // opaque, returned in completion
};

struct jinf_io_completion {
    void*   user_data;
    int32_t result;       // bytes read, or negative errno
};

// ---- Opaque context ----

struct jinf_io_context;

// ---- API ----

jinf_status jinf_io_create(jinf_io_context** ctx, const jinf_io_config* config);
void        jinf_io_destroy(jinf_io_context* ctx);

jinf_status jinf_io_open(jinf_io_context* ctx, const char* path, int* fd);
void        jinf_io_close(jinf_io_context* ctx, int fd);

// Submit one or more async read requests
jinf_status jinf_io_submit(jinf_io_context* ctx, const jinf_io_request* reqs, int count);

// Wait for completions: blocks until at least `min` completions ready, returns up to `max`
jinf_status jinf_io_wait(jinf_io_context* ctx, jinf_io_completion* comps, int min, int max, int* actual);

// Convenience: synchronous read (submit + wait)
jinf_status jinf_io_read_sync(jinf_io_context* ctx, int fd, void* buf, size_t offset, size_t length);

// Allocate 4K-aligned buffer suitable for O_DIRECT
void* jinf_io_alloc_aligned(size_t size);
void  jinf_io_free_aligned(void* ptr);
