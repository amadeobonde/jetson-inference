#include "cuda_utils.cuh"

// ---- Basic Multi-Head Attention (MHA / GQA) ----
//
// For MVP: simple loop over cached KV pairs.
// Flash attention optimization deferred to Phase 3.
//
// Input:
//   q:       [n_heads * head_dim]  — query vector for current position
//   k_cache: [max_ctx * n_heads_kv * head_dim] — cached keys
//   v_cache: [max_ctx * n_heads_kv * head_dim] — cached values
//
// Output:
//   output:  [n_heads * head_dim]  — attention output
//
// With GQA: multiple query heads share one KV head.
//   kv_head = query_head / (n_heads / n_heads_kv)

// ---- Per-head attention kernel ----
// One block per query head. Each block computes:
//   scores[t] = q_h . k_cache[t, kv_h] / sqrt(head_dim)  for t = 0..n_past-1
//   softmax(scores)
//   output_h = sum_t(scores[t] * v_cache[t, kv_h])

__global__ void kernel_attention(
    float* __restrict__ output,          // [n_heads * head_dim]
    const float* __restrict__ q,         // [n_heads * head_dim]
    const float* __restrict__ k_cache,   // [max_ctx * n_heads_kv * head_dim]
    const float* __restrict__ v_cache,   // [max_ctx * n_heads_kv * head_dim]
    int head_dim,
    int n_heads,
    int n_heads_kv,
    int n_past,      // number of tokens in cache (including current)
    int max_ctx
) {
    int head = blockIdx.x;
    if (head >= n_heads) return;

    int kv_head = head / (n_heads / n_heads_kv);

    // Shared memory for scores
    extern __shared__ float shared[];
    float* scores = shared;  // [n_past]

    const float* q_head = q + head * head_dim;
    float scale = rsqrtf((float)head_dim);

    // Phase 1: Compute attention scores
    for (int t = threadIdx.x; t < n_past; t += blockDim.x) {
        const float* k_t = k_cache + (size_t)t * n_heads_kv * head_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_head[d] * k_t[d];
        }
        scores[t] = dot * scale;
    }
    __syncthreads();

    // Phase 2: Softmax over scores
    // Find max
    float max_val = -INFINITY;
    for (int t = threadIdx.x; t < n_past; t += blockDim.x) {
        max_val = fmaxf(max_val, scores[t]);
    }
    max_val = block_reduce_max(max_val);

    __shared__ float s_max, s_sum;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();
    max_val = s_max;

    // Compute exp and sum
    float sum = 0.0f;
    for (int t = threadIdx.x; t < n_past; t += blockDim.x) {
        scores[t] = expf(scores[t] - max_val);
        sum += scores[t];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) s_sum = sum;
    __syncthreads();

    // Normalize
    float inv_sum = 1.0f / s_sum;
    for (int t = threadIdx.x; t < n_past; t += blockDim.x) {
        scores[t] *= inv_sum;
    }
    __syncthreads();

    // Phase 3: Weighted sum of values
    float* out_head = output + head * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < n_past; t++) {
            const float* v_t = v_cache + (size_t)t * n_heads_kv * head_dim + kv_head * head_dim;
            acc += scores[t] * v_t[d];
        }
        out_head[d] = acc;
    }
}

// ---- Launch wrapper ----

extern "C" void jinf_cuda_attention(float* output, const float* q, const float* k_cache,
                                     const float* v_cache, int head_dim, int n_heads,
                                     int n_heads_kv, int n_past, int max_ctx,
                                     cudaStream_t stream) {
    if (n_past <= 0) return;

    int threads = 256;
    // Shared memory: scores array for n_past positions
    size_t shared_size = n_past * sizeof(float);

    kernel_attention<<<n_heads, threads, shared_size, stream>>>(
        output, q, k_cache, v_cache, head_dim, n_heads, n_heads_kv, n_past, max_ctx);
}
