#pragma once

#include "jinf/common.h"
#include "jinf/gguf.h"
#include <cstdint>
#include <cstddef>

// ---- Weight placement ----

enum jinf_weight_placement : int {
    JINF_PLACE_HOT  = 0,   // GPU resident
    JINF_PLACE_COLD = 1,   // NVMe streamed
};

// ---- Configuration ----

struct jinf_split_config {
    size_t gpu_memory_budget;   // bytes available for hot weights
    size_t kv_cache_budget;     // bytes reserved for KV cache
    size_t buffer_budget;       // bytes reserved for I/O buffers
};

// ---- Per-tensor assignment ----

struct jinf_weight_assignment {
    const jinf_gguf_tensor_info* tensor;
    jinf_weight_placement        placement;
    int                          layer_index;  // -1 for non-layer tensors
};

// ---- Split plan ----

struct jinf_split_plan {
    jinf_weight_assignment* assignments;
    int                     count;
    size_t                  total_hot_bytes;
    size_t                  total_cold_bytes;
};

// ---- API ----

// Analyze model tensors and produce a split plan.
jinf_status jinf_split_analyze(const jinf_gguf_file* model,
                                const jinf_split_config* config,
                                jinf_split_plan** plan);

void jinf_split_plan_free(jinf_split_plan* plan);

// Print a summary of the split plan to stderr.
void jinf_split_plan_print(const jinf_split_plan* plan);
