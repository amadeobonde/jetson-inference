#include "cuda_utils.cuh"

// ---- Online Softmax with Numerical Stability ----
// 1. Find max value
// 2. Compute sum of exp(x - max)
// 3. Normalize: output[i] = exp(input[i] - max) / sum

__global__ void kernel_softmax(
    float* __restrict__ output,
    const float* __restrict__ input,
    int n
) {
    // Phase 1: Find max
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        max_val = fmaxf(max_val, input[i]);
    }
    max_val = block_reduce_max(max_val);

    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();
    max_val = s_max;

    // Phase 2: Compute sum of exp(x - max)
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum += expf(input[i] - max_val);
    }
    sum = block_reduce_sum(sum);

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = sum;
    __syncthreads();
    sum = s_sum;

    // Phase 3: Normalize
    float inv_sum = 1.0f / sum;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        output[i] = expf(input[i] - max_val) * inv_sum;
    }
}

// ---- Launch wrapper ----

extern "C" void jinf_cuda_softmax(float* output, const float* input, int n, cudaStream_t stream) {
    int threads = min(n, 1024);
    threads = ((threads + 31) / 32) * 32;
    if (threads > 1024) threads = 1024;
    kernel_softmax<<<1, threads, 0, stream>>>(output, input, n);
}
