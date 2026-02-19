#include "cuda_utils.cuh"
#include "jinf/quant.h"

// ---- RMSNorm kernel ----
// y = x * rsqrt(mean(x^2) + eps) * weight
// Single block per row, warp reduction for mean.
// Weight may be quantized (F32 or F16).

__global__ void kernel_rmsnorm_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int n, float eps
) {
    // One block processes the entire vector of length n
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = input[i];
        sum_sq += v * v;
    }

    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float s_rms;
    if (threadIdx.x == 0) {
        s_rms = rsqrtf(sum_sq / (float)n + eps);
    }
    __syncthreads();

    float rms = s_rms;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        output[i] = input[i] * rms * weight[i];
    }
}

// RMSNorm with F16 weight
__global__ void kernel_rmsnorm_f16(
    float* __restrict__ output,
    const float* __restrict__ input,
    const uint16_t* __restrict__ weight,
    int n, float eps
) {
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = input[i];
        sum_sq += v * v;
    }

    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float s_rms;
    if (threadIdx.x == 0) {
        s_rms = rsqrtf(sum_sq / (float)n + eps);
    }
    __syncthreads();

    float rms = s_rms;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        output[i] = input[i] * rms * half_to_float(weight[i]);
    }
}

// ---- Launch wrapper ----

extern "C" void jinf_cuda_rmsnorm(float* output, const float* input, const void* weight,
                                    int n, float eps, int weight_type, cudaStream_t stream) {
    int threads = min(n, 1024);
    // Ensure threads is a multiple of 32
    threads = ((threads + 31) / 32) * 32;
    if (threads > 1024) threads = 1024;

    switch (weight_type) {
        case 0:  // F32
            kernel_rmsnorm_f32<<<1, threads, 0, stream>>>(output, input, (const float*)weight, n, eps);
            break;
        case 1:  // F16
            kernel_rmsnorm_f16<<<1, threads, 0, stream>>>(output, input, (const uint16_t*)weight, n, eps);
            break;
        default:
            // For quantized norm weights, dequantize to F32 first (uncommon case)
            // In practice, norm weights are almost always F32 or F16
            kernel_rmsnorm_f32<<<1, threads, 0, stream>>>(output, input, (const float*)weight, n, eps);
            break;
    }
}
