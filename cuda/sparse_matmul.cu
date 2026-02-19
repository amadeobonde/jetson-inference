#include "cuda_utils.cuh"
#include "jinf/quant.h"

// ---- Phase 2: Sparse FFN Forward ----
//
// Processes only activated neurons. Each "neuron bundle" contains:
//   gate_row[i], up_row[i], down_col[i] — for neuron i
// bundled contiguously and padded to 16KB.
//
// Two passes:
//   1. Hot pass: neurons in GPU memory
//   2. Cold pass: neurons from NVMe buffer
// Results accumulated into output vector.

// Bundle layout per neuron (Q4_K, n_embd=8192):
//   gate_row:   8192 elements in Q4_K = 8192/256 * 144 = 4608 bytes
//   up_row:     8192 elements in Q4_K = 4608 bytes
//   down_col:   8192 elements in Q4_K = 4608 bytes
//   Total:      13824 bytes, padded to 16384 (16KB)

#define NEURON_BUNDLE_SIZE 16384  // bytes, padded

// ---- Sparse gate+up per activated neuron ----
// For each activated neuron, compute:
//   gate_val = dot(gate_row, input)
//   up_val   = dot(up_row, input)
//   activated = silu(gate_val) * up_val
//
// Then for the down projection:
//   output[d] += activated * down_col[d]

__global__ void kernel_sparse_ffn_gate_up_q4_K(
    const uint8_t* __restrict__ bundles,    // packed neuron bundles
    const float* __restrict__ input,        // [hidden_dim]
    float* __restrict__ gate_up_out,        // [num_activated * 2]
    const int* __restrict__ neuron_ids,
    const int* __restrict__ bundle_offsets,  // byte offset per activated neuron
    int num_activated,
    int hidden_dim
) {
    int act_idx = blockIdx.x;
    if (act_idx >= num_activated) return;

    int bundle_off = bundle_offsets[act_idx];
    int row_bytes = (hidden_dim / 256) * 144;  // Q4_K row size

    const jinf_block_q4_K* gate_blocks = (const jinf_block_q4_K*)(bundles + bundle_off);
    const jinf_block_q4_K* up_blocks = (const jinf_block_q4_K*)(bundles + bundle_off + row_bytes);

    int n_blocks = hidden_dim / 256;

    // Compute gate dot product
    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    for (int b = threadIdx.x / 32; b < n_blocks; b += blockDim.x / 32) {
        int lane = threadIdx.x % 32;

        // Gate block
        {
            const jinf_block_q4_K* blk = &gate_blocks[b];
            float d = half_to_float(blk->d);
            float dmin = half_to_float(blk->dmin);

            uint8_t sc[8], mn[8];
            for (int i = 0; i < 4; i++) {
                sc[i]     = blk->scales[i] & 0x3F;
                mn[i]     = blk->scales[i + 4] & 0x3F;
                sc[i + 4] = ((blk->scales[i + 8] & 0x0F) << 2) | (blk->scales[i] >> 6);
                mn[i + 4] = ((blk->scales[i + 8] >> 4)   << 2) | (blk->scales[i + 4] >> 6);
            }

            int elem_start = lane * 8;
            if (elem_start < 256) {
                int sb = elem_start / 32;
                float sub_d = d * (float)sc[sb];
                float sub_min = dmin * (float)mn[sb];
                int qs_off = elem_start / 2;
                int base = b * 256 + elem_start;
                for (int i = 0; i < 4; i++) {
                    uint8_t byte = blk->qs[qs_off + i];
                    float w0 = sub_d * (float)(byte & 0x0F) - sub_min;
                    float w1 = sub_d * (float)(byte >> 4)    - sub_min;
                    gate_sum += w0 * input[base + i * 2 + 0];
                    gate_sum += w1 * input[base + i * 2 + 1];
                }
            }
        }

        // Up block
        {
            const jinf_block_q4_K* blk = &up_blocks[b];
            float d = half_to_float(blk->d);
            float dmin = half_to_float(blk->dmin);

            uint8_t sc[8], mn[8];
            for (int i = 0; i < 4; i++) {
                sc[i]     = blk->scales[i] & 0x3F;
                mn[i]     = blk->scales[i + 4] & 0x3F;
                sc[i + 4] = ((blk->scales[i + 8] & 0x0F) << 2) | (blk->scales[i] >> 6);
                mn[i + 4] = ((blk->scales[i + 8] >> 4)   << 2) | (blk->scales[i + 4] >> 6);
            }

            int elem_start = lane * 8;
            if (elem_start < 256) {
                int sb = elem_start / 32;
                float sub_d = d * (float)sc[sb];
                float sub_min = dmin * (float)mn[sb];
                int qs_off = elem_start / 2;
                int base = b * 256 + elem_start;
                for (int i = 0; i < 4; i++) {
                    uint8_t byte = blk->qs[qs_off + i];
                    float w0 = sub_d * (float)(byte & 0x0F) - sub_min;
                    float w1 = sub_d * (float)(byte >> 4)    - sub_min;
                    up_sum += w0 * input[base + i * 2 + 0];
                    up_sum += w1 * input[base + i * 2 + 1];
                }
            }
        }
    }

    gate_sum = block_reduce_sum(gate_sum);
    up_sum = block_reduce_sum(up_sum);

    if (threadIdx.x == 0) {
        // SiLU(gate) * up
        float silu = gate_sum / (1.0f + expf(-gate_sum));
        gate_up_out[act_idx] = silu * up_sum;
    }
}

// ---- Sparse down projection: scatter-add activated values ----
// output[d] += sum_i(activated[i] * down_col_i[d])

__global__ void kernel_sparse_ffn_down_q4_K(
    const uint8_t* __restrict__ bundles,
    float* __restrict__ output,             // [hidden_dim]
    const float* __restrict__ activated,    // [num_activated] — SiLU(gate)*up values
    const int* __restrict__ bundle_offsets,
    int num_activated,
    int hidden_dim
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= hidden_dim) return;

    int row_bytes = (hidden_dim / 256) * 144;

    float acc = 0.0f;
    for (int act_idx = 0; act_idx < num_activated; act_idx++) {
        int bundle_off = bundle_offsets[act_idx];
        // Down column starts after gate_row + up_row
        const jinf_block_q4_K* down_blocks =
            (const jinf_block_q4_K*)(bundles + bundle_off + 2 * row_bytes);

        // Dequantize element d from down column
        int block_idx = d / 256;
        int elem_in_block = d % 256;
        const jinf_block_q4_K* blk = &down_blocks[block_idx];

        float dd = half_to_float(blk->d);
        float dmin = half_to_float(blk->dmin);

        int sb = elem_in_block / 32;
        uint8_t sc_val, mn_val;
        if (sb < 4) {
            sc_val = blk->scales[sb] & 0x3F;
            mn_val = blk->scales[sb + 4] & 0x3F;
        } else {
            sc_val = ((blk->scales[sb - 4 + 8] & 0x0F) << 2) | (blk->scales[sb - 4] >> 6);
            mn_val = ((blk->scales[sb - 4 + 8] >> 4) << 2) | (blk->scales[sb - 4 + 4] >> 6);
        }

        float sub_d = dd * (float)sc_val;
        float sub_min = dmin * (float)mn_val;

        int qs_idx = elem_in_block / 2;
        uint8_t byte = blk->qs[qs_idx];
        float w;
        if (elem_in_block % 2 == 0) {
            w = sub_d * (float)(byte & 0x0F) - sub_min;
        } else {
            w = sub_d * (float)(byte >> 4) - sub_min;
        }

        acc += activated[act_idx] * w;
    }

    atomicAdd(&output[d], acc);
}

// ---- Combined sparse FFN launch ----

extern "C" void jinf_cuda_sparse_ffn(
    const void* bundles,
    const float* input,
    float* output,
    const int* neuron_ids,
    const int* bundle_offsets,
    int num_activated,
    int hidden_dim,
    cudaStream_t stream
) {
    if (num_activated == 0) return;

    // Allocate temporary for gate_up results
    float* gate_up_tmp;
    cudaMalloc(&gate_up_tmp, num_activated * sizeof(float));

    // Phase 1: Gate + Up + SiLU for each activated neuron
    kernel_sparse_ffn_gate_up_q4_K<<<num_activated, 256, 0, stream>>>(
        (const uint8_t*)bundles, input, gate_up_tmp,
        neuron_ids, bundle_offsets, num_activated, hidden_dim);

    // Phase 2: Down projection (scatter-add)
    // Zero output first
    cudaMemsetAsync(output, 0, hidden_dim * sizeof(float), stream);

    int threads = 256;
    int blocks = (hidden_dim + threads - 1) / threads;
    kernel_sparse_ffn_down_q4_K<<<blocks, threads, 0, stream>>>(
        (const uint8_t*)bundles, output, gate_up_tmp,
        bundle_offsets, num_activated, hidden_dim);

    cudaFree(gate_up_tmp);
}
