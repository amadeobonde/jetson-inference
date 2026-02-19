#include "jinf/buffer_pool.h"

#include <cstdlib>
#include <cstring>

jinf_status jinf_buffer_create(jinf_buffer_pair** bp, const jinf_buffer_config* config) {
    if (!bp || !config || config->capacity == 0) return JINF_ERR_INVALID;

    jinf_buffer_pair* b = (jinf_buffer_pair*)calloc(1, sizeof(jinf_buffer_pair));
    if (!b) return JINF_ERR_OOM;

    b->capacity = config->capacity;
    b->active = 0;
    b->loaded_size[0] = 0;
    b->loaded_size[1] = 0;

    // Allocate pinned memory (cudaHostAlloc gives us pinned + page-aligned memory).
    // On Jetson unified memory, this is directly GPU-accessible (zero-copy).
    for (int i = 0; i < 2; i++) {
        cudaError_t err = cudaHostAlloc(&b->host_pinned[i], config->capacity,
                                         cudaHostAllocDefault);
        if (err != cudaSuccess) {
            JINF_LOG("cudaHostAlloc(%zu) failed: %s",
                     config->capacity, cudaGetErrorString(err));
            // Cleanup what we allocated
            for (int j = 0; j < i; j++) {
                cudaFreeHost(b->host_pinned[j]);
            }
            free(b);
            return JINF_ERR_CUDA;
        }
    }

    // Create CUDA streams and events
    JINF_CUDA_CHECK(cudaStreamCreate(&b->compute_stream));
    JINF_CUDA_CHECK(cudaStreamCreate(&b->transfer_stream));
    JINF_CUDA_CHECK(cudaEventCreate(&b->compute_done));
    JINF_CUDA_CHECK(cudaEventCreate(&b->transfer_done));

    *bp = b;
    return JINF_OK;
}

void jinf_buffer_destroy(jinf_buffer_pair* bp) {
    if (!bp) return;

    cudaEventDestroy(bp->compute_done);
    cudaEventDestroy(bp->transfer_done);
    cudaStreamDestroy(bp->compute_stream);
    cudaStreamDestroy(bp->transfer_stream);

    for (int i = 0; i < 2; i++) {
        if (bp->host_pinned[i]) {
            cudaFreeHost(bp->host_pinned[i]);
        }
    }

    free(bp);
}

jinf_status jinf_buffer_start_read(jinf_buffer_pair* bp, jinf_io_context* io,
                                    int fd, size_t offset, size_t size) {
    if (!bp || !io) return JINF_ERR_INVALID;

    // Read into the inactive buffer
    int inactive = 1 - bp->active;

    if (size > bp->capacity) {
        JINF_LOG("Read size %zu exceeds buffer capacity %zu", size, bp->capacity);
        return JINF_ERR_INVALID;
    }

    // Ensure the inactive buffer isn't still being used by the GPU
    // (from a previous iteration where it was the active buffer)

    // Submit async read via io_uring into the inactive pinned buffer
    // The pinned buffer is 4K-aligned from cudaHostAlloc, satisfying O_DIRECT
    size_t aligned_size = JINF_ALIGN_4K(size);
    if (aligned_size > bp->capacity) aligned_size = bp->capacity;

    jinf_io_request req = {
        .fd = fd,
        .buffer = bp->host_pinned[inactive],
        .offset = offset,
        .length = aligned_size,
        .user_data = bp,
    };

    jinf_status s = jinf_io_submit(io, &req, 1);
    if (s != JINF_OK) return s;

    bp->loaded_size[inactive] = size;  // actual useful bytes
    return JINF_OK;
}

jinf_status jinf_buffer_wait_read(jinf_buffer_pair* bp, jinf_io_context* io,
                                   void** out_ptr, size_t* out_size) {
    if (!bp || !io) return JINF_ERR_INVALID;

    int inactive = 1 - bp->active;

    // Wait for the io_uring completion
    jinf_io_completion comp;
    int actual = 0;
    jinf_status s = jinf_io_wait(io, &comp, 1, 1, &actual);
    if (s != JINF_OK) return s;

    if (comp.result < 0) {
        JINF_LOG("Buffer read failed: %d", comp.result);
        return JINF_ERR_IO;
    }

    if (out_ptr) *out_ptr = bp->host_pinned[inactive];
    if (out_size) *out_size = bp->loaded_size[inactive];

    return JINF_OK;
}

void jinf_buffer_swap(jinf_buffer_pair* bp) {
    if (!bp) return;
    bp->active = 1 - bp->active;
}

void* jinf_buffer_active_ptr(const jinf_buffer_pair* bp) {
    return bp ? bp->host_pinned[bp->active] : nullptr;
}

size_t jinf_buffer_active_size(const jinf_buffer_pair* bp) {
    return bp ? bp->loaded_size[bp->active] : 0;
}

jinf_status jinf_buffer_signal_compute_done(jinf_buffer_pair* bp) {
    if (!bp) return JINF_ERR_INVALID;
    JINF_CUDA_CHECK(cudaEventRecord(bp->compute_done, bp->compute_stream));
    return JINF_OK;
}

jinf_status jinf_buffer_wait_compute_done(jinf_buffer_pair* bp) {
    if (!bp) return JINF_ERR_INVALID;
    JINF_CUDA_CHECK(cudaEventSynchronize(bp->compute_done));
    return JINF_OK;
}

// ---- Phase 2: Neuron bundle reads ----

jinf_status jinf_buffer_start_bundle_range_read(
    jinf_buffer_pair* bp, jinf_io_context* io, int fd,
    size_t file_offset, size_t total_size) {
    // This is essentially the same as start_read — read a contiguous range
    // into the inactive buffer. The caller computes the range covering
    // all needed bundles and reads the whole thing.
    return jinf_buffer_start_read(bp, io, fd, file_offset, total_size);
}

void* jinf_buffer_get_bundle(jinf_buffer_pair* bp, int base_neuron,
                              int target_neuron, size_t bundle_size) {
    if (!bp) return nullptr;
    int inactive = 1 - bp->active;
    size_t offset = (size_t)(target_neuron - base_neuron) * bundle_size;
    return (char*)bp->host_pinned[inactive] + offset;
}
