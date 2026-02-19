#include "cuda_utils.cuh"
#include "jinf/quant.h"

// ---- Phase 2: Sparse FFN Forward ----
//
// Processes only activated neurons. Each "neuron bundle" contains:
//   gate_row[i] (quantized), up_row[i] (quantized), down_col[i] (float32)
// bundled contiguously and padded to 4K.
//
// Two phases:
//   1. Gate+Up: For each activated neuron, compute SiLU(dot(gate_row, input)) * dot(up_row, input)
//   2. Down: output[d] += sum_i(activated[i] * down_col_i[d])

// ---- Q4_K Gate+Up kernel ----

__global__ void kernel_sparse_ffn_gate_up_q4_K(
    const uint8_t* __restrict__ bundles,
    const float* __restrict__ input,
    float* __restrict__ gate_up_out,
    const int* __restrict__ bundle_offsets,
    int num_activated,
    int hidden_dim,
    int gate_row_bytes
) {
    int act_idx = blockIdx.x;
    if (act_idx >= num_activated) return;

    int bundle_off = bundle_offsets[act_idx];

    const jinf_block_q4_K* gate_blocks = (const jinf_block_q4_K*)(bundles + bundle_off);
    const jinf_block_q4_K* up_blocks = (const jinf_block_q4_K*)(bundles + bundle_off + gate_row_bytes);

    int n_blocks = hidden_dim / 256;

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
        float silu = gate_sum / (1.0f + expf(-gate_sum));
        gate_up_out[act_idx] = silu * up_sum;
    }
}

// ---- Q6_K Gate+Up kernel ----

__global__ void kernel_sparse_ffn_gate_up_q6_K(
    const uint8_t* __restrict__ bundles,
    const float* __restrict__ input,
    float* __restrict__ gate_up_out,
    const int* __restrict__ bundle_offsets,
    int num_activated,
    int hidden_dim,
    int gate_row_bytes
) {
    int act_idx = blockIdx.x;
    if (act_idx >= num_activated) return;

    int bundle_off = bundle_offsets[act_idx];

    const jinf_block_q6_K* gate_blocks = (const jinf_block_q6_K*)(bundles + bundle_off);
    const jinf_block_q6_K* up_blocks = (const jinf_block_q6_K*)(bundles + bundle_off + gate_row_bytes);

    int n_blocks = hidden_dim / 256;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    for (int b = threadIdx.x / 32; b < n_blocks; b += blockDim.x / 32) {
        int lane = threadIdx.x % 32;

        // Each lane processes 8 elements
        int elem_start = lane * 8;
        if (elem_start >= 256) continue;

        // Gate block
        {
            const jinf_block_q6_K* blk = &gate_blocks[b];
            float d = half_to_float(blk->d);
            int sb = elem_start / 16;
            float scale = d * (float)blk->scales[sb];
            int base = b * 256 + elem_start;

            for (int i = 0; i < 8; i++) {
                int idx = elem_start + i;
                // Extract 6-bit value: low 4 bits from ql, high 2 bits from qh
                uint8_t ql_byte = blk->ql[idx / 2];
                uint8_t low4 = (idx % 2 == 0) ? (ql_byte & 0x0F) : (ql_byte >> 4);
                uint8_t qh_byte = blk->qh[idx / 4];
                uint8_t high2 = (qh_byte >> (2 * (idx % 4))) & 0x03;
                int8_t q6 = (int8_t)((low4 | (high2 << 4)) - 32);
                float w = scale * (float)q6;
                gate_sum += w * input[base + i];
            }
        }

        // Up block
        {
            const jinf_block_q6_K* blk = &up_blocks[b];
            float d = half_to_float(blk->d);
            int sb = elem_start / 16;
            float scale = d * (float)blk->scales[sb];
            int base = b * 256 + elem_start;

            for (int i = 0; i < 8; i++) {
                int idx = elem_start + i;
                uint8_t ql_byte = blk->ql[idx / 2];
                uint8_t low4 = (idx % 2 == 0) ? (ql_byte & 0x0F) : (ql_byte >> 4);
                uint8_t qh_byte = blk->qh[idx / 4];
                uint8_t high2 = (qh_byte >> (2 * (idx % 4))) & 0x03;
                int8_t q6 = (int8_t)((low4 | (high2 << 4)) - 32);
                float w = scale * (float)q6;
                up_sum += w * input[base + i];
            }
        }
    }

    gate_sum = block_reduce_sum(gate_sum);
    up_sum = block_reduce_sum(up_sum);

    if (threadIdx.x == 0) {
        float silu = gate_sum / (1.0f + expf(-gate_sum));
        gate_up_out[act_idx] = silu * up_sum;
    }
}

// ---- Down projection kernel (float32 down_col) ----
// output[d] += sum_i(activated[i] * down_col_i[d])
// Down columns are stored as float32 in the bundle (after gate_row + up_row).

__global__ void kernel_sparse_ffn_down_f32(
    const uint8_t* __restrict__ bundles,
    float* __restrict__ output,
    const float* __restrict__ activated,
    const int* __restrict__ bundle_offsets,
    int num_activated,
    int hidden_dim,
    int down_col_offset   // byte offset of down_col within bundle
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= hidden_dim) return;

    float acc = 0.0f;
    for (int act_idx = 0; act_idx < num_activated; act_idx++) {
        int bundle_off = bundle_offsets[act_idx];
        const float* down_col = (const float*)(bundles + bundle_off + down_col_offset);
        acc += activated[act_idx] * down_col[d];
    }

    atomicAdd(&output[d], acc);
}

// ---- Combined sparse FFN launch ----

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
    cudaStream_t stream
) {
    if (num_activated == 0) return;

    // Compute row bytes based on qtype
    int block_size, type_bytes;
    switch (qtype) {
        case 12: // JINF_TYPE_Q4_K
            block_size = 256; type_bytes = 144; break;
        case 14: // JINF_TYPE_Q6_K
            block_size = 256; type_bytes = 210; break;
        default:
            // Fallback to Q4_K
            block_size = 256; type_bytes = 144; break;
    }

    int gate_row_bytes = (hidden_dim / block_size) * type_bytes;

    // Phase 1: Gate + Up + SiLU for each activated neuron
    switch (qtype) {
        case 12: // Q4_K
            kernel_sparse_ffn_gate_up_q4_K<<<num_activated, 256, 0, stream>>>(
                (const uint8_t*)bundles, input, gate_up_tmp,
                bundle_offsets, num_activated, hidden_dim, gate_row_bytes);
            break;
        case 14: // Q6_K
            kernel_sparse_ffn_gate_up_q6_K<<<num_activated, 256, 0, stream>>>(
                (const uint8_t*)bundles, input, gate_up_tmp,
                bundle_offsets, num_activated, hidden_dim, gate_row_bytes);
            break;
        default:
            kernel_sparse_ffn_gate_up_q4_K<<<num_activated, 256, 0, stream>>>(
                (const uint8_t*)bundles, input, gate_up_tmp,
                bundle_offsets, num_activated, hidden_dim, gate_row_bytes);
            break;
    }

    // Phase 2: Down projection (float32 down columns, scatter-add)
    cudaMemsetAsync(output, 0, hidden_dim * sizeof(float), stream);

    // Down column offset within bundle = gate_row_bytes + up_row_bytes
    // up_row has the same qtype and row_bytes as gate
    int down_col_offset = 2 * gate_row_bytes;

    int threads = 256;
    int blocks = (hidden_dim + threads - 1) / threads;
    kernel_sparse_ffn_down_f32<<<blocks, threads, 0, stream>>>(
        (const uint8_t*)bundles, output, gate_up_tmp,
        bundle_offsets, num_activated, hidden_dim, down_col_offset);
}
