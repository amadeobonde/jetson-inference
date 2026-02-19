#include "cuda_utils.cuh"

// ---- Rotary Position Embedding (RoPE) ----
// Applied in-place to Q and K tensors.
// For each pair of elements (q[2i], q[2i+1]), rotate by angle theta_i * pos.
// theta_i = freq_base ^ (-2i / head_dim)

__global__ void kernel_rope(
    float* __restrict__ q,      // [n_heads * head_dim]
    float* __restrict__ k,      // [n_heads_kv * head_dim]
    int head_dim,
    int n_heads,
    int n_heads_kv,
    int pos,
    float freq_base
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total_q_pairs = n_heads * half_dim;
    int total_k_pairs = n_heads_kv * half_dim;
    int total_pairs = total_q_pairs + total_k_pairs;

    if (idx >= total_pairs) return;

    float* data;
    int pair_idx;

    if (idx < total_q_pairs) {
        data = q;
        pair_idx = idx;
    } else {
        data = k;
        pair_idx = idx - total_q_pairs;
    }

    int head = pair_idx / half_dim;
    int i    = pair_idx % half_dim;

    float theta = powf(freq_base, -2.0f * (float)i / (float)head_dim);
    float angle = (float)pos * theta;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    int offset = head * head_dim + i * 2;
    float x0 = data[offset + 0];
    float x1 = data[offset + 1];

    data[offset + 0] = x0 * cos_a - x1 * sin_a;
    data[offset + 1] = x0 * sin_a + x1 * cos_a;
}

// ---- Launch wrapper ----

extern "C" void jinf_cuda_rope(float* q, float* k, int head_dim, int n_heads, int n_heads_kv,
                                int pos, float freq_base, cudaStream_t stream) {
    int half_dim = head_dim / 2;
    int total_pairs = n_heads * half_dim + n_heads_kv * half_dim;
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;
    kernel_rope<<<blocks, threads, 0, stream>>>(q, k, head_dim, n_heads, n_heads_kv, pos, freq_base);
}
