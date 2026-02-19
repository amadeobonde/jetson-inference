#include "cuda_utils.cuh"
#include "jinf/quant.h"

// ---- Fused dequant + mat-vec for Q4_0 ----
// y[row] = dot(dequant(W[row, :]), x[:])
// W is [rows x cols] stored in Q4_0 blocks along cols dimension

__global__ void kernel_dequant_matvec_q4_0(
    const jinf_block_q4_0* __restrict__ weight,
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    // Number of Q4_0 blocks per row
    int n_blocks_per_row = cols / 32;

    // Each thread handles some blocks
    float sum = 0.0f;
    for (int b = threadIdx.x; b < n_blocks_per_row; b += blockDim.x) {
        const jinf_block_q4_0* blk = &weight[row * n_blocks_per_row + b];
        float d = half_to_float(blk->d);
        int base = b * 32;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t byte = blk->qs[i];
            float v0 = ((float)((int)(byte & 0x0F) - 8)) * d;
            float v1 = ((float)((int)(byte >> 4) - 8))    * d;
            sum += v0 * input[base + i * 2 + 0];
            sum += v1 * input[base + i * 2 + 1];
        }
    }

    // Warp reduction then block reduction
    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        output[row] = sum;
    }
}

// ---- Fused dequant + mat-vec for Q4_K ----

__global__ void kernel_dequant_matvec_q4_K(
    const jinf_block_q4_K* __restrict__ weight,
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int n_blocks_per_row = cols / 256;

    float sum = 0.0f;
    for (int b = threadIdx.x / 32; b < n_blocks_per_row; b += blockDim.x / 32) {
        const jinf_block_q4_K* blk = &weight[row * n_blocks_per_row + b];
        float d    = half_to_float(blk->d);
        float dmin = half_to_float(blk->dmin);

        // Decode scales (all threads in the warp compute this)
        uint8_t sc[8], mn[8];
        for (int i = 0; i < 4; i++) {
            sc[i]     = blk->scales[i] & 0x3F;
            mn[i]     = blk->scales[i + 4] & 0x3F;
            sc[i + 4] = ((blk->scales[i + 8] & 0x0F) << 2) | (blk->scales[i] >> 6);
            mn[i + 4] = ((blk->scales[i + 8] >> 4)   << 2) | (blk->scales[i + 4] >> 6);
        }

        int lane = threadIdx.x % 32;
        int base_elem = b * 256;

        // Each thread in the warp handles 8 elements
        int elem_start = lane * 8;
        if (elem_start < 256) {
            int sb = elem_start / 32;
            float sub_d   = d * (float)sc[sb];
            float sub_min = dmin * (float)mn[sb];

            int qs_off = elem_start / 2;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint8_t byte = blk->qs[qs_off + i];
                float w0 = sub_d * (float)(byte & 0x0F) - sub_min;
                float w1 = sub_d * (float)(byte >> 4)    - sub_min;
                sum += w0 * input[base_elem + elem_start + i * 2 + 0];
                sum += w1 * input[base_elem + elem_start + i * 2 + 1];
            }
        }
    }

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        output[row] = sum;
    }
}

// ---- Fused dequant + mat-vec for Q8_0 ----

__global__ void kernel_dequant_matvec_q8_0(
    const jinf_block_q8_0* __restrict__ weight,
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int n_blocks_per_row = cols / 32;

    float sum = 0.0f;
    for (int b = threadIdx.x; b < n_blocks_per_row; b += blockDim.x) {
        const jinf_block_q8_0* blk = &weight[row * n_blocks_per_row + b];
        float d = half_to_float(blk->d);
        int base = b * 32;

        #pragma unroll
        for (int i = 0; i < 32; i++) {
            sum += ((float)blk->qs[i] * d) * input[base + i];
        }
    }

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        output[row] = sum;
    }
}

// ---- Fused dequant + mat-vec for Q6_K ----

__global__ void kernel_dequant_matvec_q6_K(
    const jinf_block_q6_K* __restrict__ weight,
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int n_blocks_per_row = cols / 256;

    float sum = 0.0f;
    for (int b = threadIdx.x / 32; b < n_blocks_per_row; b += blockDim.x / 32) {
        const jinf_block_q6_K* blk = &weight[row * n_blocks_per_row + b];
        float d = half_to_float(blk->d);

        int lane = threadIdx.x % 32;
        int base_elem = b * 256;

        // Each thread in the warp handles 8 elements
        int elem_start = lane * 8;
        if (elem_start < 256) {
            int sb = elem_start / 16;  // sub-block index
            float scale = d * (float)blk->scales[sb];

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = elem_start + i;

                // Lower 4 bits from ql
                uint8_t ql_byte = blk->ql[idx / 2];
                int low4 = (idx & 1) ? (ql_byte >> 4) : (ql_byte & 0x0F);

                // Upper 2 bits from qh
                uint8_t qh_byte = blk->qh[idx / 4];
                int high2 = (qh_byte >> (2 * (idx % 4))) & 0x03;

                int q6 = low4 | (high2 << 4);
                float w = scale * (float)(q6 - 32);
                sum += w * input[base_elem + idx];
            }
        }
    }

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        output[row] = sum;
    }
}

// ---- Residual add kernel ----

__global__ void kernel_residual_add(float* __restrict__ a, const float* __restrict__ b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] += b[idx];
    }
}

// ---- Launch wrapper ----

extern "C" void jinf_cuda_dequant_matvec(const void* weight, const float* input, float* output,
                                           int rows, int cols, int qtype, cudaStream_t stream) {
    if (rows == 0 || cols == 0) return;

    int threads = 256;

    switch (qtype) {
        case JINF_TYPE_Q4_0:
            kernel_dequant_matvec_q4_0<<<rows, threads, 0, stream>>>(
                (const jinf_block_q4_0*)weight, input, output, rows, cols);
            break;
        case JINF_TYPE_Q4_K:
            kernel_dequant_matvec_q4_K<<<rows, threads, 0, stream>>>(
                (const jinf_block_q4_K*)weight, input, output, rows, cols);
            break;
        case JINF_TYPE_Q8_0:
            kernel_dequant_matvec_q8_0<<<rows, threads, 0, stream>>>(
                (const jinf_block_q8_0*)weight, input, output, rows, cols);
            break;
        case JINF_TYPE_Q6_K:
            kernel_dequant_matvec_q6_K<<<rows, threads, 0, stream>>>(
                (const jinf_block_q6_K*)weight, input, output, rows, cols);
            break;
        default:
            fprintf(stderr, "[jinf] Unsupported qtype %d for dequant_matvec\n", qtype);
            break;
    }
}

extern "C" void jinf_cuda_residual_add(float* a, const float* b, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_residual_add<<<blocks, threads, 0, stream>>>(a, b, n);
}
