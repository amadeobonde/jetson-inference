#pragma once

#include "jinf/common.h"
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

// ---- Phase 2: Sparse MatMul Interface ----
//
// Performs FFN computation using only activated neurons (gate/up/down),
// reading from a mix of hot (GPU-resident) and cold (NVMe-streamed) bundles.

struct jinf_sparse_ffn_args {
    const void*  neuron_bundles;     // packed cold neuron bundles from NVMe buffer
    const void*  hot_bundles;        // hot neuron weights (GPU resident)
    const float* input;              // hidden state [n_embd]
    float*       output;             // output [n_embd]
    const int*   neuron_ids;         // which neurons are activated (device mem)
    const int*   is_hot;             // 1=hot, 0=cold per activated neuron (device mem)
    const int*   cold_offsets;       // byte offset into neuron_bundles for each cold neuron
    int          num_activated;
    int          hidden_dim;         // n_embd
    int          qtype;              // jinf_qtype of the weights
    cudaStream_t stream;
};

// Launch sparse FFN forward pass on GPU.
jinf_status jinf_sparse_ffn_forward(const jinf_sparse_ffn_args* args);

// Dense mat-vec (for non-sparse layers like attention): y = x @ W^T
// W is quantized [rows x cols], x is float [cols], y is float [rows].
jinf_status jinf_dense_matvec(const void* weight, const float* input, float* output,
                               int rows, int cols, int qtype, cudaStream_t stream);
