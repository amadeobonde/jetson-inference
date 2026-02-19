#pragma once

#include "jinf/common.h"
#include "jinf/nvme_io.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// ---- Double-buffer pipeline ----
//
// Two pinned memory buffers allocated via cudaHostAlloc (pinned + 4K-aligned).
// While GPU computes on buffer[active], io_uring fills buffer[1-active] with the
// next layer's cold weights. On Jetson unified memory, pinned buffers are directly
// GPU-accessible (zero-copy) — no explicit H2D cudaMemcpy needed.

struct jinf_buffer_pair {
    void*         host_pinned[2];     // cudaHostAlloc buffers
    size_t        capacity;           // per-buffer capacity in bytes
    int           active;             // 0 or 1: which buffer GPU reads from
    size_t        loaded_size[2];     // actual bytes loaded in each buffer
    cudaStream_t  compute_stream;
    cudaStream_t  transfer_stream;
    cudaEvent_t   compute_done;
    cudaEvent_t   transfer_done;
};

struct jinf_buffer_config {
    size_t capacity;   // per-buffer capacity (e.g., 512MB for 70B)
};

// ---- API ----

jinf_status jinf_buffer_create(jinf_buffer_pair** bp, const jinf_buffer_config* config);
void        jinf_buffer_destroy(jinf_buffer_pair* bp);

// Start an async NVMe read into the inactive buffer.
jinf_status jinf_buffer_start_read(jinf_buffer_pair* bp, jinf_io_context* io,
                                    int fd, size_t offset, size_t size);

// Wait for the pending read to complete. Returns pointer to the now-filled buffer.
jinf_status jinf_buffer_wait_read(jinf_buffer_pair* bp, jinf_io_context* io,
                                   void** out_ptr, size_t* out_size);

// Swap active/inactive indices (call after GPU is done with current active buffer).
void jinf_buffer_swap(jinf_buffer_pair* bp);

// Get a pointer to the active buffer (the one GPU should read from).
void* jinf_buffer_active_ptr(const jinf_buffer_pair* bp);
size_t jinf_buffer_active_size(const jinf_buffer_pair* bp);

// Signal that GPU compute on the active buffer is done.
jinf_status jinf_buffer_signal_compute_done(jinf_buffer_pair* bp);

// Wait until GPU compute on the active buffer is done (before overwriting it).
jinf_status jinf_buffer_wait_compute_done(jinf_buffer_pair* bp);

// ---- Phase 2: Neuron bundle reads ----

// Read a contiguous range of neuron bundles into the inactive buffer.
// file_offset: absolute file offset of the first bundle in range
// total_size:  total bytes to read (covers the range from min to max neuron)
// Uses the simpler contiguous-read approach (all bundles are sequential in file).
jinf_status jinf_buffer_start_bundle_range_read(
    jinf_buffer_pair* bp, jinf_io_context* io, int fd,
    size_t file_offset, size_t total_size);

// After wait_read completes, get a pointer to a specific bundle within the buffer.
// base_neuron: the first neuron ID that was read (the range start)
// target_neuron: the neuron ID to get
// bundle_size: bytes per bundle
void* jinf_buffer_get_bundle(jinf_buffer_pair* bp, int base_neuron,
                              int target_neuron, size_t bundle_size);
