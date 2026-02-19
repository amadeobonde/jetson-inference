#pragma once

#include "jinf/common.h"
#include "jinf/nvmw.h"
#include "jinf/nvme_io.h"
#include "jinf/buffer_pool.h"
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

// ---- Performance stats ----

struct jinf_perf_stats {
    double total_ms;
    double nvme_read_ms;
    double gpu_compute_ms;
    double attention_ms;
    double ffn_ms;
    int    tokens_generated;
    double nvme_bytes_read;
};

// ---- Engine configuration ----

struct jinf_engine_config {
    const char* model_path;       // path to .nvmw file
    size_t      gpu_memory_budget; // total GPU memory budget
    size_t      buffer_capacity;   // per-buffer capacity for double buffering
    int         max_context;       // max sequence length
    int         io_queue_depth;    // io_uring queue depth
};

static inline jinf_engine_config jinf_engine_config_default() {
    return {
        .model_path       = nullptr,
        .gpu_memory_budget = (size_t)4500 * 1024 * 1024,
        .buffer_capacity   = (size_t)512 * 1024 * 1024,
        .max_context       = 2048,
        .io_queue_depth    = 64,
    };
}

// ---- Per-layer weight pointers ----

struct jinf_layer_ptrs {
    // Norms (always hot)
    void* attn_norm;     // RMSNorm weight [n_embd]
    void* ffn_norm;      // RMSNorm weight [n_embd]

    // Attention weights
    void* attn_q;        // [n_embd, n_embd]
    void* attn_k;        // [n_embd, n_heads_kv * head_dim]
    void* attn_v;        // [n_embd, n_heads_kv * head_dim]
    void* attn_output;   // [n_embd, n_embd]

    // FFN weights
    void* ffn_gate;      // [n_embd, n_ff]
    void* ffn_up;        // [n_embd, n_ff]
    void* ffn_down;      // [n_ff, n_embd]

    bool is_hot;         // all weights for this layer are GPU-resident
};

// ---- Engine state ----

struct jinf_engine {
    // Model data
    jinf_nvmw_reader*  model;
    jinf_io_context*   io;
    jinf_buffer_pair*  buffers;
    int                fd_model;

    // Hot weights (one big GPU allocation)
    void* hot_weights_gpu;
    size_t hot_weights_size;

    // KV cache [n_layers][max_context][n_heads_kv * head_dim]
    float* kv_cache_k;
    float* kv_cache_v;

    // Per-layer pointers
    jinf_layer_ptrs* layers;

    // Scratch buffers
    float* scratch_a;
    float* scratch_b;
    float* logits_buf;

    // State
    int n_past;           // current KV cache position

    // CUDA
    cudaStream_t compute_stream;

    // Model parameters
    int n_layers;
    int n_embd;
    int n_heads;
    int n_heads_kv;
    int n_ff;
    int n_vocab;
    int max_context;
    int head_dim;
    float rms_norm_eps;
    float rope_freq_base;
    int32_t primary_type;  // jinf_qtype

    // Stats
    jinf_perf_stats perf;
};

// ---- API ----

jinf_status jinf_engine_create(jinf_engine** e, const jinf_engine_config* config);
void        jinf_engine_destroy(jinf_engine* e);

// Run forward pass for one or more tokens. Returns pointer to logits [n_vocab].
jinf_status jinf_engine_forward(jinf_engine* e, const int32_t* tokens, int n_tokens,
                                 float** logits);

// Generate tokens autoregressively with greedy sampling.
jinf_status jinf_engine_generate(jinf_engine* e, const int32_t* prompt, int n_prompt,
                                  int32_t* output, int max_tokens, int* n_generated);

// Reset KV cache and position.
void jinf_engine_reset(jinf_engine* e);

// Get performance stats.
const jinf_perf_stats* jinf_engine_get_stats(const jinf_engine* e);
