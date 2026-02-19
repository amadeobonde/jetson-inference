#pragma once

#include "jinf/common.h"
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

// ---- Phase 2: Sparse FFN Interface ----
//
// Performs FFN computation using only activated neurons (gate/up/down),
// reading from neuron bundles (same format for hot and cold).
//
// Bundle layout per neuron:
//   [gate_row: quantized n_embd values]
//   [up_row: quantized n_embd values]
//   [down_col: float32 n_embd values]  (dequantized during model prep)
// Padded to 4K alignment.

// Sparse FFN forward: processes only activated neurons.
// bundles:        packed neuron bundles (GPU-accessible memory)
// input:          hidden state [hidden_dim]
// output:         output vector [hidden_dim] (will be zeroed, then accumulated)
// bundle_offsets: byte offset within `bundles` for each activated neuron [num_activated]
// num_activated:  number of active neurons
// hidden_dim:     n_embd
// bundle_size:    padded bytes per neuron bundle (runtime, not compiled-in)
// qtype:          quantization type of gate/up rows (down is always float32)
// gate_up_tmp:    pre-allocated GPU buffer [num_activated] for intermediate values
// stream:         CUDA stream
extern "C" void jinf_cuda_sparse_ffn(
    const void* bundles,
    const float* input,
    float* output,
    const int* bundle_offsets,
    int num_activated,
    int hidden_dim,
    int bundle_size,
    int qtype,
    float* gate_up_tmp,
    cudaStream_t stream);
