#include "cuda_utils.cuh"

// ---- Fused SiLU(gate) * up ----
// output[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]

__global__ void kernel_silu_mul(
    float* __restrict__ output,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = gate[idx];
    float silu = g / (1.0f + expf(-g));
    output[idx] = silu * up[idx];
}

// ---- Launch wrapper ----

extern "C" void jinf_cuda_silu_mul(float* output, const float* gate, const float* up,
                                    int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_silu_mul<<<blocks, threads, 0, stream>>>(output, gate, up, n);
}
