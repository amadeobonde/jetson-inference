#include "cuda_utils.cuh"
#include "jinf/quant.h"

// ---- Q4_0 dequantization: 1 thread per block of 32 values ----

__global__ void kernel_dequantize_q4_0(const jinf_block_q4_0* __restrict__ blocks,
                                        float* __restrict__ output,
                                        int n_blocks) {
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= n_blocks) return;

    float d = half_to_float(blocks[bid].d);
    float* out = output + bid * 32;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint8_t byte = blocks[bid].qs[i];
        out[i * 2 + 0] = ((float)((int)(byte & 0x0F) - 8)) * d;
        out[i * 2 + 1] = ((float)((int)(byte >> 4) - 8))    * d;
    }
}

// ---- Q4_K dequantization: 1 warp (32 threads) per super-block of 256 values ----

__global__ void kernel_dequantize_q4_K(const jinf_block_q4_K* __restrict__ blocks,
                                        float* __restrict__ output,
                                        int n_blocks) {
    int bid = blockIdx.x;
    if (bid >= n_blocks) return;

    int tid = threadIdx.x;  // 0..31

    float d    = half_to_float(blocks[bid].d);
    float dmin = half_to_float(blocks[bid].dmin);

    // Decode 6-bit sub-block scales and minimums
    __shared__ uint8_t sc[8], mn[8];
    if (tid < 4) {
        sc[tid]     = blocks[bid].scales[tid] & 0x3F;
        mn[tid]     = blocks[bid].scales[tid + 4] & 0x3F;
        sc[tid + 4] = ((blocks[bid].scales[tid + 8] & 0x0F) << 2) | (blocks[bid].scales[tid] >> 6);
        mn[tid + 4] = ((blocks[bid].scales[tid + 8] >> 4)   << 2) | (blocks[bid].scales[tid + 4] >> 6);
    }
    __syncthreads();

    float* out = output + bid * 256;

    // Each thread handles 8 values (32 threads x 8 = 256 values)
    int base = tid * 8;
    int sb = base / 32;  // sub-block index
    float sub_d   = d * (float)sc[sb];
    float sub_min = dmin * (float)mn[sb];

    int qs_off = base / 2;  // each byte = 2 values
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint8_t byte = blocks[bid].qs[qs_off + i];
        out[base + i * 2 + 0] = sub_d * (float)(byte & 0x0F) - sub_min;
        out[base + i * 2 + 1] = sub_d * (float)(byte >> 4)    - sub_min;
    }
}

// ---- Q8_0 dequantization: 1 thread per block of 32 values ----

__global__ void kernel_dequantize_q8_0(const jinf_block_q8_0* __restrict__ blocks,
                                        float* __restrict__ output,
                                        int n_blocks) {
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= n_blocks) return;

    float d = half_to_float(blocks[bid].d);
    float* out = output + bid * 32;

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        out[i] = (float)blocks[bid].qs[i] * d;
    }
}

// ---- Q6_K dequantization: 1 warp (32 threads) per super-block of 256 values ----

__global__ void kernel_dequantize_q6_K(const jinf_block_q6_K* __restrict__ blocks,
                                        float* __restrict__ output,
                                        int n_blocks) {
    int bid = blockIdx.x;
    if (bid >= n_blocks) return;

    int tid = threadIdx.x;  // 0..31

    float d = half_to_float(blocks[bid].d);
    float* out = output + bid * 256;

    // Each thread handles 8 contiguous values
    int elem_start = tid * 8;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = elem_start + i;
        int sb = idx / 16;  // sub-block index (0..15)

        // Lower 4 bits from ql (2 values per byte)
        uint8_t ql_byte = blocks[bid].ql[idx / 2];
        int low4 = (idx & 1) ? (ql_byte >> 4) : (ql_byte & 0x0F);

        // Upper 2 bits from qh (4 values per byte)
        uint8_t qh_byte = blocks[bid].qh[idx / 4];
        int high2 = (qh_byte >> (2 * (idx % 4))) & 0x03;

        int q6 = low4 | (high2 << 4);  // 6-bit value (0..63)
        out[idx] = d * (float)blocks[bid].scales[sb] * (float)(q6 - 32);
    }
}

// ---- Launch wrappers ----

extern "C" void jinf_cuda_dequantize_q4_0(const void* data, float* output,
                                            int n_blocks, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n_blocks + threads - 1) / threads;
    kernel_dequantize_q4_0<<<blocks, threads, 0, stream>>>(
        (const jinf_block_q4_0*)data, output, n_blocks);
}

extern "C" void jinf_cuda_dequantize_q4_K(const void* data, float* output,
                                            int n_blocks, cudaStream_t stream) {
    // 1 block per Q4_K super-block, 32 threads per block (1 warp)
    kernel_dequantize_q4_K<<<n_blocks, 32, 0, stream>>>(
        (const jinf_block_q4_K*)data, output, n_blocks);
}

extern "C" void jinf_cuda_dequantize_q8_0(const void* data, float* output,
                                            int n_blocks, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n_blocks + threads - 1) / threads;
    kernel_dequantize_q8_0<<<blocks, threads, 0, stream>>>(
        (const jinf_block_q8_0*)data, output, n_blocks);
}

extern "C" void jinf_cuda_dequantize_q6_K(const void* data, float* output,
                                            int n_blocks, cudaStream_t stream) {
    // 1 block per Q6_K super-block, 32 threads per block (1 warp)
    kernel_dequantize_q6_K<<<n_blocks, 32, 0, stream>>>(
        (const jinf_block_q6_K*)data, output, n_blocks);
}
