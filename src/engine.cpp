#include "jinf/engine.h"
#include "jinf/quant.h"
#include "jinf/sparse_ops.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

// ---- CUDA kernel declarations (defined in cuda/) ----

extern "C" {
void jinf_cuda_rmsnorm(float* output, const float* input, const void* weight,
                        int n, float eps, int weight_type, cudaStream_t stream);
void jinf_cuda_rope(float* q, float* k, int head_dim, int n_heads, int n_heads_kv,
                     int pos, float freq_base, cudaStream_t stream);
void jinf_cuda_silu_mul(float* output, const float* gate, const float* up,
                         int n, cudaStream_t stream);
void jinf_cuda_softmax(float* output, const float* input, int n, cudaStream_t stream);
void jinf_cuda_dequant_matvec(const void* weight, const float* input, float* output,
                               int rows, int cols, int qtype, cudaStream_t stream);
void jinf_cuda_attention(float* output, const float* q, const float* k_cache,
                          const float* v_cache, int head_dim, int n_heads,
                          int n_heads_kv, int n_past, int max_ctx,
                          cudaStream_t stream);
void jinf_cuda_dequantize_q4_0(const void* data, float* output, int n_blocks, cudaStream_t stream);
void jinf_cuda_dequantize_q4_K(const void* data, float* output, int n_blocks, cudaStream_t stream);
void jinf_cuda_dequantize_q8_0(const void* data, float* output, int n_blocks, cudaStream_t stream);
void jinf_cuda_dequantize_q6_K(const void* data, float* output, int n_blocks, cudaStream_t stream);
void jinf_cuda_residual_add(float* a, const float* b, int n, cudaStream_t stream);
}

// ---- Helpers ----

static size_t kv_cache_size(int n_layers, int max_ctx, int n_heads_kv, int head_dim) {
    return (size_t)n_layers * max_ctx * n_heads_kv * head_dim * sizeof(float);
}

// ---- Engine creation ----

jinf_status jinf_engine_create(jinf_engine** e, const jinf_engine_config* config) {
    if (!e || !config || !config->model_path) return JINF_ERR_INVALID;

    jinf_engine* eng = (jinf_engine*)calloc(1, sizeof(jinf_engine));
    if (!eng) return JINF_ERR_OOM;

    // Open .nvmw model
    jinf_status s = jinf_nvmw_open(&eng->model, config->model_path);
    if (s != JINF_OK) { free(eng); return s; }

    const jinf_nvmw_header* hdr = jinf_nvmw_get_header(eng->model);

    // Copy model parameters
    eng->n_layers     = (int)hdr->n_layers;
    eng->n_embd       = (int)hdr->n_embd;
    eng->n_heads      = (int)hdr->n_heads;
    eng->n_heads_kv   = (int)hdr->n_heads_kv;
    eng->n_ff         = (int)hdr->n_ff;
    eng->n_vocab      = (int)hdr->n_vocab;
    eng->max_context  = config->max_context;
    eng->head_dim     = eng->n_embd / eng->n_heads;
    eng->rms_norm_eps = hdr->rms_norm_eps;
    eng->rope_freq_base = hdr->rope_freq_base;
    eng->primary_type = hdr->primary_type;
    eng->n_past       = 0;

    JINF_LOG("Engine: %d layers, embd=%d, heads=%d/%d, ff=%d, vocab=%d, head_dim=%d, ctx=%d",
             eng->n_layers, eng->n_embd, eng->n_heads, eng->n_heads_kv,
             eng->n_ff, eng->n_vocab, eng->head_dim, eng->max_context);

    // Initialize CUDA context early
    JINF_CUDA_CHECK(cudaSetDevice(0));
    JINF_CUDA_CHECK(cudaFree(0));

    // Create io_uring context
    jinf_io_config io_cfg = {
        .queue_depth = config->io_queue_depth,
        .read_size = 1024 * 1024,
        .flags = 0,
    };
    s = jinf_io_create(&eng->io, &io_cfg);
    if (s != JINF_OK) { jinf_engine_destroy(eng); return s; }

    // Open model file for O_DIRECT reads
    s = jinf_io_open(eng->io, config->model_path, &eng->fd_model);
    if (s != JINF_OK) { jinf_engine_destroy(eng); return s; }

    // Create double-buffer pipeline
    jinf_buffer_config buf_cfg = { .capacity = config->buffer_capacity };
    s = jinf_buffer_create(&eng->buffers, &buf_cfg);
    if (s != JINF_OK) { jinf_engine_destroy(eng); return s; }

    // Allocate and load hot weights
    uint64_t hot_offset, hot_size;
    s = jinf_nvmw_get_hot_range(eng->model, &hot_offset, &hot_size);
    if (s != JINF_OK) { jinf_engine_destroy(eng); return s; }

    eng->hot_weights_size = (size_t)hot_size;
    JINF_LOG("Loading hot weights: %.1f MB", hot_size / (1024.0 * 1024.0));

    JINF_CUDA_CHECK(cudaMalloc(&eng->hot_weights_gpu, hot_size));

    // Load hot weights in chunks via a small staging buffer to avoid
    // doubling memory usage (critical on Jetson unified memory).
    static const size_t STAGING_SIZE = 4 * 1024 * 1024; // 4 MB
    void* staging = nullptr;
    JINF_CUDA_CHECK(cudaHostAlloc(&staging, STAGING_SIZE, cudaHostAllocDefault));

    size_t remaining = (size_t)hot_size;
    size_t off = 0;
    while (remaining > 0) {
        size_t chunk = std::min(remaining, STAGING_SIZE);
        size_t aligned_chunk = JINF_ALIGN_4K(chunk);
        s = jinf_io_read_sync(eng->io, eng->fd_model, staging,
                               hot_offset + off, aligned_chunk);
        if (s != JINF_OK) {
            cudaFreeHost(staging);
            jinf_engine_destroy(eng);
            return s;
        }
        JINF_CUDA_CHECK(cudaMemcpy((char*)eng->hot_weights_gpu + off,
                                    staging, chunk, cudaMemcpyHostToDevice));
        off += chunk;
        remaining -= chunk;
    }
    cudaFreeHost(staging);

    // Build per-layer pointers and types
    eng->layers = (jinf_layer_ptrs*)calloc(eng->n_layers, sizeof(jinf_layer_ptrs));
    if (!eng->layers) { jinf_engine_destroy(eng); return JINF_ERR_OOM; }

    int n_entries = 0;
    const jinf_nvmw_tensor_entry* entries = jinf_nvmw_get_tensor_entries(eng->model, &n_entries);

    for (int i = 0; i < n_entries; i++) {
        const jinf_nvmw_tensor_entry* te = &entries[i];
        int layer = te->layer_index;
        if (layer < 0 || layer >= eng->n_layers) continue;

        void* base_ptr;
        if (te->is_hot) {
            base_ptr = (char*)eng->hot_weights_gpu + te->offset;
        } else {
            continue; // Cold weights loaded at runtime
        }

        jinf_layer_ptrs* lp = &eng->layers[layer];
        lp->is_hot = true;

        if (strstr(te->name, "attn_norm.weight"))         { lp->attn_norm = base_ptr;   lp->attn_norm_type = te->type; }
        else if (strstr(te->name, "ffn_norm.weight"))      { lp->ffn_norm = base_ptr;    lp->ffn_norm_type = te->type; }
        else if (strstr(te->name, "attn_q.weight"))        { lp->attn_q = base_ptr;      lp->attn_q_type = te->type; }
        else if (strstr(te->name, "attn_k.weight"))        { lp->attn_k = base_ptr;      lp->attn_k_type = te->type; }
        else if (strstr(te->name, "attn_v.weight"))        { lp->attn_v = base_ptr;      lp->attn_v_type = te->type; }
        else if (strstr(te->name, "attn_output.weight"))   { lp->attn_output = base_ptr; lp->attn_output_type = te->type; }
        else if (strstr(te->name, "ffn_gate.weight"))      { lp->ffn_gate = base_ptr;    lp->ffn_gate_type = te->type; }
        else if (strstr(te->name, "ffn_up.weight"))        { lp->ffn_up = base_ptr;      lp->ffn_up_type = te->type; }
        else if (strstr(te->name, "ffn_down.weight"))      { lp->ffn_down = base_ptr;    lp->ffn_down_type = te->type; }
    }

    // Mark layers with cold tensors
    for (int i = 0; i < n_entries; i++) {
        const jinf_nvmw_tensor_entry* te = &entries[i];
        if (!te->is_hot && te->layer_index >= 0 && te->layer_index < eng->n_layers) {
            eng->layers[te->layer_index].is_hot = false;
        }
    }

    // Cache embedding tensor info (avoid per-token lookup)
    const jinf_nvmw_tensor_entry* embd = jinf_nvmw_find_tensor(eng->model, "token_embd.weight");
    if (embd) {
        eng->embed_type = embd->type;
        jinf_qtype qt = (jinf_qtype)embd->type;
        int block_size = jinf_qtype_block_size(qt);
        eng->embed_n_blocks = (eng->n_embd + block_size - 1) / block_size;
        eng->embed_row_bytes = (size_t)eng->embed_n_blocks * jinf_qtype_type_size(qt);
        eng->embed_weight = (char*)eng->hot_weights_gpu + embd->offset;
    }

    // Cache output norm and output weight tensor info
    const jinf_nvmw_tensor_entry* onorm = jinf_nvmw_find_tensor(eng->model, "output_norm.weight");
    if (onorm) {
        eng->output_norm_weight = (char*)eng->hot_weights_gpu + onorm->offset;
        eng->output_norm_type = onorm->type;
    }
    const jinf_nvmw_tensor_entry* oweight = jinf_nvmw_find_tensor(eng->model, "output.weight");
    if (oweight) {
        eng->output_weight = (char*)eng->hot_weights_gpu + oweight->offset;
        eng->output_weight_type = oweight->type;
    }

    // Allocate KV cache
    size_t kv_size = kv_cache_size(eng->n_layers, eng->max_context,
                                    eng->n_heads_kv, eng->head_dim);
    JINF_LOG("KV cache: %.1f MB per K/V", kv_size / (1024.0 * 1024.0));
    JINF_CUDA_CHECK(cudaMalloc((void**)&eng->kv_cache_k, kv_size));
    JINF_CUDA_CHECK(cudaMalloc((void**)&eng->kv_cache_v, kv_size));
    JINF_CUDA_CHECK(cudaMemset(eng->kv_cache_k, 0, kv_size));
    JINF_CUDA_CHECK(cudaMemset(eng->kv_cache_v, 0, kv_size));

    // Allocate scratch buffers
    size_t scratch_size = std::max({
        (size_t)eng->n_embd,
        (size_t)eng->n_ff,
        (size_t)eng->n_vocab
    }) * sizeof(float);
    JINF_CUDA_CHECK(cudaMalloc((void**)&eng->scratch_a, scratch_size));
    JINF_CUDA_CHECK(cudaMalloc((void**)&eng->scratch_b, scratch_size));
    JINF_CUDA_CHECK(cudaMalloc((void**)&eng->logits_buf, eng->n_vocab * sizeof(float)));
    JINF_CUDA_CHECK(cudaMalloc((void**)&eng->hidden_buf, eng->n_embd * sizeof(float)));

    // Pre-allocate pinned host buffer for greedy sampling (avoids malloc/free per token)
    JINF_CUDA_CHECK(cudaHostAlloc((void**)&eng->host_logits,
                                   eng->n_vocab * sizeof(float), cudaHostAllocDefault));

    // Create compute stream
    JINF_CUDA_CHECK(cudaStreamCreate(&eng->compute_stream));

    // Phase 2: Initialize sparse FFN if predictor path is provided
    eng->sparse_enabled = false;
    eng->predictor = nullptr;
    eng->sparse_gate_up_tmp = nullptr;
    eng->host_hidden_state = nullptr;
    eng->host_active_ids = nullptr;
    eng->dev_bundle_offsets = nullptr;
    eng->bundle_size = 0;

    if (config->predictor_path && jinf_nvmw_has_bundles(eng->model)) {
        s = jinf_predictor_create(&eng->predictor, config->predictor_path,
                                   config->sparsity_threshold);
        if (s == JINF_OK) {
            // Allocate sparse FFN buffers
            JINF_CUDA_CHECK(cudaMalloc((void**)&eng->sparse_gate_up_tmp,
                                        eng->n_ff * sizeof(float)));
            JINF_CUDA_CHECK(cudaHostAlloc((void**)&eng->host_hidden_state,
                                           eng->n_embd * sizeof(float), cudaHostAllocDefault));

            eng->host_active_ids = (int*)malloc(eng->n_ff * sizeof(int));
            JINF_CUDA_CHECK(cudaMalloc((void**)&eng->dev_bundle_offsets,
                                        eng->n_ff * sizeof(int)));

            const jinf_nvmw_bundle_header* bh = jinf_nvmw_get_bundle_header(eng->model);
            eng->bundle_size = bh ? bh->bundle_size : 0;

            eng->sparse_enabled = (eng->bundle_size > 0);

            if (eng->sparse_enabled) {
                JINF_LOG("Sparse FFN enabled: bundle_size=%zu", eng->bundle_size);
            }
        } else {
            JINF_LOG("Warning: Failed to load predictor, using dense FFN");
        }
    } else if (config->predictor_path) {
        JINF_LOG("Warning: Predictor path set but model has no bundles, using dense FFN");
    }

    memset(&eng->perf, 0, sizeof(eng->perf));

    *e = eng;
    return JINF_OK;
}

void jinf_engine_destroy(jinf_engine* e) {
    if (!e) return;

    // Phase 2: cleanup sparse FFN
    if (e->predictor) jinf_predictor_destroy(e->predictor);
    if (e->sparse_gate_up_tmp) cudaFree(e->sparse_gate_up_tmp);
    if (e->host_hidden_state) cudaFreeHost(e->host_hidden_state);
    free(e->host_active_ids);
    if (e->dev_bundle_offsets) cudaFree(e->dev_bundle_offsets);

    if (e->compute_stream) cudaStreamDestroy(e->compute_stream);
    if (e->host_logits) cudaFreeHost(e->host_logits);
    if (e->hidden_buf) cudaFree(e->hidden_buf);
    if (e->logits_buf) cudaFree(e->logits_buf);
    if (e->scratch_b) cudaFree(e->scratch_b);
    if (e->scratch_a) cudaFree(e->scratch_a);
    if (e->kv_cache_v) cudaFree(e->kv_cache_v);
    if (e->kv_cache_k) cudaFree(e->kv_cache_k);
    if (e->hot_weights_gpu) cudaFree(e->hot_weights_gpu);

    free(e->layers);
    if (e->buffers) jinf_buffer_destroy(e->buffers);
    if (e->io) {
        if (e->fd_model >= 0) jinf_io_close(e->io, e->fd_model);
        jinf_io_destroy(e->io);
    }
    if (e->model) jinf_nvmw_close(e->model);

    free(e);
}

void jinf_engine_reset(jinf_engine* e) {
    if (!e) return;
    e->n_past = 0;
    size_t kv_size = kv_cache_size(e->n_layers, e->max_context,
                                    e->n_heads_kv, e->head_dim);
    cudaMemset(e->kv_cache_k, 0, kv_size);
    cudaMemset(e->kv_cache_v, 0, kv_size);
    memset(&e->perf, 0, sizeof(e->perf));
}

const jinf_perf_stats* jinf_engine_get_stats(const jinf_engine* e) {
    return e ? &e->perf : nullptr;
}

// ---- Forward pass helpers ----

// Get embedding for a token from hot weights (uses cached tensor info)
static void embed_token(jinf_engine* e, int32_t token_id, float* out) {
    const void* row_ptr = (const char*)e->embed_weight + (size_t)token_id * e->embed_row_bytes;

    switch (e->embed_type) {
        case JINF_TYPE_Q4_0:
            jinf_cuda_dequantize_q4_0(row_ptr, out, e->embed_n_blocks, e->compute_stream);
            break;
        case JINF_TYPE_Q4_K:
            jinf_cuda_dequantize_q4_K(row_ptr, out, e->embed_n_blocks, e->compute_stream);
            break;
        case JINF_TYPE_Q8_0:
            jinf_cuda_dequantize_q8_0(row_ptr, out, e->embed_n_blocks, e->compute_stream);
            break;
        case JINF_TYPE_Q6_K:
            jinf_cuda_dequantize_q6_K(row_ptr, out, e->embed_n_blocks, e->compute_stream);
            break;
        case JINF_TYPE_F32:
            cudaMemcpyAsync(out, row_ptr, e->n_embd * sizeof(float),
                            cudaMemcpyDeviceToDevice, e->compute_stream);
            break;
        default:
            break;
    }
}

// ---- Phase 2: Sparse FFN for one layer ----
// Called after attention is done. Replaces the dense FFN steps (8-10).
// Works for both hot and cold layers — the bundle data is passed as a GPU-accessible pointer.
static jinf_status forward_layer_sparse_ffn(jinf_engine* e, int layer, float* hidden_state,
                                              const void* bundle_data, int gate_qtype) {
    int n = e->n_embd;
    int n_ff = e->n_ff;
    cudaStream_t stream = e->compute_stream;
    jinf_layer_ptrs* lp = &e->layers[layer];

    float* norm_out = e->scratch_a;

    // 8. FFN RMSNorm
    jinf_cuda_rmsnorm(norm_out, hidden_state, lp->ffn_norm, n, e->rms_norm_eps, lp->ffn_norm_type, stream);

    // Predict active neurons (CPU side)
    // Copy norm_out to host for predictor (needed for future MLP predictor)
    cudaStreamSynchronize(stream);
    cudaMemcpy(e->host_hidden_state, norm_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    int n_active = 0;
    jinf_status s = jinf_predictor_predict(e->predictor, e->host_hidden_state, layer,
                                            e->host_active_ids, &n_active);
    if (s != JINF_OK || n_active == 0) {
        // Fallback: no neurons active, just skip FFN (output is just the residual)
        return JINF_OK;
    }

    // Build bundle offsets array on host, then copy to device
    // Each active neuron's bundle is at: (neuron_id * bundle_size) bytes into bundle_data
    // But bundle_data may be offset from the layer start. For hot bundles, it's the
    // absolute GPU pointer to the layer's bundle region. For cold bundles read into
    // a buffer, bundle_data points to the start of the read range.
    // We compute offsets relative to bundle_data.

    // For the simple contiguous read approach:
    // We need to know the base neuron that was read. For the hot case, base = 0
    // (all bundles are in GPU memory). For cold, we read a contiguous range
    // covering min_active..max_active, so base = min_active.

    int min_active = e->host_active_ids[0];
    int max_active = e->host_active_ids[0];
    for (int i = 1; i < n_active; i++) {
        if (e->host_active_ids[i] < min_active) min_active = e->host_active_ids[i];
        if (e->host_active_ids[i] > max_active) max_active = e->host_active_ids[i];
    }

    // The bundle offsets are relative to bundle_data pointer
    // For hot: bundle_data already points to neuron 0's bundle
    // For cold: bundle_data points to min_active's bundle (start of the read range)
    int base_neuron = 0; // hot case
    // Check if this is a cold layer — cold layers have bundle_data pointing to
    // a buffer starting at min_active, not neuron 0
    if (!lp->is_hot) {
        base_neuron = min_active;
    }

    // Build host-side offset array
    int* host_offsets = (int*)e->host_active_ids; // reuse buffer temporarily? No, we still need IDs.
    // Use a separate stack allocation since n_active is bounded by n_ff (~11k), ~44KB on stack is fine.
    int* host_bundle_offsets = (int*)alloca(n_active * sizeof(int));
    for (int i = 0; i < n_active; i++) {
        host_bundle_offsets[i] = (e->host_active_ids[i] - base_neuron) * (int)e->bundle_size;
    }

    // Copy offsets to device
    cudaMemcpy(e->dev_bundle_offsets, host_bundle_offsets, n_active * sizeof(int), cudaMemcpyHostToDevice);

    // Launch sparse FFN kernel
    float* down_out = norm_out; // reuse scratch_a for output
    jinf_cuda_sparse_ffn(bundle_data, norm_out, down_out,
                          e->dev_bundle_offsets, n_active, n,
                          (int)e->bundle_size, gate_qtype,
                          e->sparse_gate_up_tmp, stream);

    // 10. Residual add: hidden_state += down_out
    jinf_cuda_residual_add(hidden_state, down_out, n, stream);

    return JINF_OK;
}

// Process one layer with all weights hot (in GPU memory)
static jinf_status forward_layer_hot(jinf_engine* e, int layer, float* hidden_state) {
    jinf_layer_ptrs* lp = &e->layers[layer];
    int n = e->n_embd;
    int n_ff = e->n_ff;
    int head_dim = e->head_dim;
    cudaStream_t stream = e->compute_stream;

    float* norm_out = e->scratch_a;
    float* scratch = e->scratch_b;

    // 1. Attention RMSNorm
    jinf_cuda_rmsnorm(norm_out, hidden_state, lp->attn_norm, n, e->rms_norm_eps, lp->attn_norm_type, stream);

    // 2. Q, K, V projections (each may have different quantization type)
    float* q_buf = scratch;
    float* k_buf = q_buf + n;
    int kv_dim = e->n_heads_kv * head_dim;
    float* v_buf = k_buf + kv_dim;

    jinf_cuda_dequant_matvec(lp->attn_q, norm_out, q_buf, n, n, lp->attn_q_type, stream);
    jinf_cuda_dequant_matvec(lp->attn_k, norm_out, k_buf, kv_dim, n, lp->attn_k_type, stream);
    jinf_cuda_dequant_matvec(lp->attn_v, norm_out, v_buf, kv_dim, n, lp->attn_v_type, stream);

    // 3. RoPE
    jinf_cuda_rope(q_buf, k_buf, head_dim, e->n_heads, e->n_heads_kv,
                    e->n_past, e->rope_freq_base, stream);

    // 4. Store K, V into cache
    size_t kv_layer_stride = (size_t)e->max_context * kv_dim;
    float* k_cache_layer = e->kv_cache_k + layer * kv_layer_stride;
    float* v_cache_layer = e->kv_cache_v + layer * kv_layer_stride;

    cudaMemcpyAsync(k_cache_layer + (size_t)e->n_past * kv_dim, k_buf,
                     kv_dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(v_cache_layer + (size_t)e->n_past * kv_dim, v_buf,
                     kv_dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // 5. Attention: Q @ K^T, softmax, @ V
    float* attn_out = norm_out;
    jinf_cuda_attention(attn_out, q_buf, k_cache_layer, v_cache_layer,
                         head_dim, e->n_heads, e->n_heads_kv,
                         e->n_past + 1, e->max_context, stream);

    // 6. Output projection
    float* proj_out = scratch;
    jinf_cuda_dequant_matvec(lp->attn_output, attn_out, proj_out, n, n, lp->attn_output_type, stream);

    // 7. Residual add: hidden_state += proj_out
    jinf_cuda_residual_add(hidden_state, proj_out, n, stream);

    // 8-10. FFN (sparse or dense)
    if (e->sparse_enabled && jinf_nvmw_has_bundles(e->model)) {
        // Sparse FFN: use neuron bundles from hot weights
        // Hot bundle data for this layer: look up the layer's bundle region in GPU memory
        const jinf_nvmw_bundle_layer_desc* bld = jinf_nvmw_get_bundle_layer_desc(e->model, layer);
        if (bld && bld->bundles_offset > 0) {
            // For hot layers, we need the bundle data in GPU memory.
            // The bundles are appended after the cold region in the NVMW file.
            // For hot layers, we need to pre-load bundles into GPU memory too.
            // For MVP: read bundles from NVMe into the active buffer synchronously.
            size_t range_size = (size_t)e->n_ff * e->bundle_size;
            void* buf = jinf_buffer_active_ptr(e->buffers);
            jinf_io_read_sync(e->io, e->fd_model, buf, bld->bundles_offset,
                              JINF_ALIGN_4K(range_size));
            return forward_layer_sparse_ffn(e, layer, hidden_state, buf, lp->ffn_gate_type);
        }
    }

    // Dense FFN fallback
    jinf_cuda_rmsnorm(norm_out, hidden_state, lp->ffn_norm, n, e->rms_norm_eps, lp->ffn_norm_type, stream);

    float* gate_buf = scratch;
    float* up_buf = gate_buf + n_ff;

    jinf_cuda_dequant_matvec(lp->ffn_gate, norm_out, gate_buf, n_ff, n, lp->ffn_gate_type, stream);

    // FFN profiling hook: capture raw gate output before SiLU
    if (e->ffn_hook) {
        cudaStreamSynchronize(stream);
        e->ffn_hook(e->ffn_hook_data, layer, gate_buf, n_ff);
    }

    jinf_cuda_dequant_matvec(lp->ffn_up, norm_out, up_buf, n_ff, n, lp->ffn_up_type, stream);

    jinf_cuda_silu_mul(gate_buf, gate_buf, up_buf, n_ff, stream);

    float* down_out = norm_out;
    jinf_cuda_dequant_matvec(lp->ffn_down, gate_buf, down_out, n, n_ff, lp->ffn_down_type, stream);

    // 10. Residual add: hidden_state += down_out
    jinf_cuda_residual_add(hidden_state, down_out, n, stream);

    return JINF_OK;
}

// Process one cold layer: read from NVMe buffer, compute
static jinf_status forward_layer_cold(jinf_engine* e, int layer, float* hidden_state,
                                       void* cold_buffer) {
    jinf_layer_ptrs* lp = &e->layers[layer];
    int n = e->n_embd;
    int n_ff = e->n_ff;
    int head_dim = e->head_dim;
    cudaStream_t stream = e->compute_stream;

    float* norm_out = e->scratch_a;
    float* scratch = e->scratch_b;

    // Parse cold buffer: find offset and type of each tensor
    int n_entries = 0;
    const jinf_nvmw_tensor_entry* entries = jinf_nvmw_get_tensor_entries(e->model, &n_entries);

    void* cold_attn_q = nullptr;       int cold_attn_q_type = 0;
    void* cold_attn_k = nullptr;       int cold_attn_k_type = 0;
    void* cold_attn_v = nullptr;       int cold_attn_v_type = 0;
    void* cold_attn_output = nullptr;  int cold_attn_output_type = 0;
    void* cold_ffn_gate = nullptr;     int cold_ffn_gate_type = 0;
    void* cold_ffn_up = nullptr;       int cold_ffn_up_type = 0;
    void* cold_ffn_down = nullptr;     int cold_ffn_down_type = 0;

    for (int i = 0; i < n_entries; i++) {
        const jinf_nvmw_tensor_entry* te = &entries[i];
        if (te->is_hot || te->layer_index != layer) continue;

        void* ptr = (char*)cold_buffer + te->offset;

        if (strstr(te->name, "attn_q.weight"))             { cold_attn_q = ptr;      cold_attn_q_type = te->type; }
        else if (strstr(te->name, "attn_k.weight"))         { cold_attn_k = ptr;      cold_attn_k_type = te->type; }
        else if (strstr(te->name, "attn_v.weight"))         { cold_attn_v = ptr;      cold_attn_v_type = te->type; }
        else if (strstr(te->name, "attn_output.weight"))    { cold_attn_output = ptr; cold_attn_output_type = te->type; }
        else if (strstr(te->name, "ffn_gate.weight"))       { cold_ffn_gate = ptr;    cold_ffn_gate_type = te->type; }
        else if (strstr(te->name, "ffn_up.weight"))         { cold_ffn_up = ptr;      cold_ffn_up_type = te->type; }
        else if (strstr(te->name, "ffn_down.weight"))       { cold_ffn_down = ptr;    cold_ffn_down_type = te->type; }
    }

    // Use hot pointers/types for norms (always hot), cold for the rest
    void* attn_q = cold_attn_q ? cold_attn_q : lp->attn_q;
    void* attn_k = cold_attn_k ? cold_attn_k : lp->attn_k;
    void* attn_v = cold_attn_v ? cold_attn_v : lp->attn_v;
    void* attn_output = cold_attn_output ? cold_attn_output : lp->attn_output;
    void* ffn_gate = cold_ffn_gate ? cold_ffn_gate : lp->ffn_gate;
    void* ffn_up = cold_ffn_up ? cold_ffn_up : lp->ffn_up;
    void* ffn_down = cold_ffn_down ? cold_ffn_down : lp->ffn_down;

    int qt_q   = cold_attn_q      ? cold_attn_q_type      : lp->attn_q_type;
    int qt_k   = cold_attn_k      ? cold_attn_k_type      : lp->attn_k_type;
    int qt_v   = cold_attn_v      ? cold_attn_v_type      : lp->attn_v_type;
    int qt_out = cold_attn_output ? cold_attn_output_type  : lp->attn_output_type;
    int qt_gate = cold_ffn_gate   ? cold_ffn_gate_type     : lp->ffn_gate_type;
    int qt_up   = cold_ffn_up     ? cold_ffn_up_type       : lp->ffn_up_type;
    int qt_down = cold_ffn_down   ? cold_ffn_down_type     : lp->ffn_down_type;

    // 1. Attention RMSNorm (norm weight is always hot)
    jinf_cuda_rmsnorm(norm_out, hidden_state, lp->attn_norm, n, e->rms_norm_eps, lp->attn_norm_type, stream);

    // 2-6. Attention
    float* q_buf = scratch;
    float* k_buf = q_buf + n;
    int kv_dim = e->n_heads_kv * head_dim;
    float* v_buf = k_buf + kv_dim;

    jinf_cuda_dequant_matvec(attn_q, norm_out, q_buf, n, n, qt_q, stream);
    jinf_cuda_dequant_matvec(attn_k, norm_out, k_buf, kv_dim, n, qt_k, stream);
    jinf_cuda_dequant_matvec(attn_v, norm_out, v_buf, kv_dim, n, qt_v, stream);

    jinf_cuda_rope(q_buf, k_buf, head_dim, e->n_heads, e->n_heads_kv,
                    e->n_past, e->rope_freq_base, stream);

    size_t kv_layer_stride = (size_t)e->max_context * kv_dim;
    float* k_cache_layer = e->kv_cache_k + layer * kv_layer_stride;
    float* v_cache_layer = e->kv_cache_v + layer * kv_layer_stride;

    cudaMemcpyAsync(k_cache_layer + (size_t)e->n_past * kv_dim, k_buf,
                     kv_dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(v_cache_layer + (size_t)e->n_past * kv_dim, v_buf,
                     kv_dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    float* attn_out = norm_out;
    jinf_cuda_attention(attn_out, q_buf, k_cache_layer, v_cache_layer,
                         head_dim, e->n_heads, e->n_heads_kv,
                         e->n_past + 1, e->max_context, stream);

    float* proj_out = scratch;
    jinf_cuda_dequant_matvec(attn_output, attn_out, proj_out, n, n, qt_out, stream);

    jinf_cuda_residual_add(hidden_state, proj_out, n, stream);

    // 8-10. FFN (sparse or dense)
    if (e->sparse_enabled && jinf_nvmw_has_bundles(e->model)) {
        const jinf_nvmw_bundle_layer_desc* bld = jinf_nvmw_get_bundle_layer_desc(e->model, layer);
        if (bld && bld->bundles_offset > 0) {
            // Sparse FFN for cold layer: read only the needed neuron bundles
            // First, predict which neurons are active (need norm_out for future MLP predictor)
            jinf_cuda_rmsnorm(norm_out, hidden_state, lp->ffn_norm, n, e->rms_norm_eps, lp->ffn_norm_type, stream);
            cudaStreamSynchronize(stream);
            cudaMemcpy(e->host_hidden_state, norm_out, n * sizeof(float), cudaMemcpyDeviceToHost);

            int n_active = 0;
            jinf_predictor_predict(e->predictor, e->host_hidden_state, layer,
                                    e->host_active_ids, &n_active);

            if (n_active > 0) {
                // Find min/max active neuron for contiguous range read
                int min_id = e->host_active_ids[0], max_id = e->host_active_ids[0];
                for (int i = 1; i < n_active; i++) {
                    if (e->host_active_ids[i] < min_id) min_id = e->host_active_ids[i];
                    if (e->host_active_ids[i] > max_id) max_id = e->host_active_ids[i];
                }

                // Read contiguous range from min_id to max_id into buffer
                uint64_t range_start = bld->bundles_offset + (uint64_t)min_id * bld->bundle_size;
                size_t range_size = (size_t)(max_id - min_id + 1) * bld->bundle_size;
                void* buf = jinf_buffer_active_ptr(e->buffers);
                jinf_io_read_sync(e->io, e->fd_model, buf, range_start,
                                  JINF_ALIGN_4K(range_size));

                // Build bundle offsets relative to the read buffer
                int* host_bundle_offsets = (int*)alloca(n_active * sizeof(int));
                for (int i = 0; i < n_active; i++) {
                    host_bundle_offsets[i] = (e->host_active_ids[i] - min_id) * (int)e->bundle_size;
                }
                cudaMemcpy(e->dev_bundle_offsets, host_bundle_offsets,
                           n_active * sizeof(int), cudaMemcpyHostToDevice);

                // Launch sparse FFN (norm_out already computed above)
                float* down_out = norm_out;
                jinf_cuda_sparse_ffn(buf, norm_out, down_out,
                                      e->dev_bundle_offsets, n_active, n,
                                      (int)e->bundle_size, qt_gate,
                                      e->sparse_gate_up_tmp, stream);

                jinf_cuda_residual_add(hidden_state, down_out, n, stream);
                return JINF_OK;
            }
            // n_active == 0: skip FFN entirely
            return JINF_OK;
        }
    }

    // Dense FFN fallback
    jinf_cuda_rmsnorm(norm_out, hidden_state, lp->ffn_norm, n, e->rms_norm_eps, lp->ffn_norm_type, stream);

    float* gate_buf = scratch;
    float* up_buf = gate_buf + n_ff;

    jinf_cuda_dequant_matvec(ffn_gate, norm_out, gate_buf, n_ff, n, qt_gate, stream);

    // FFN profiling hook: capture raw gate output before SiLU
    if (e->ffn_hook) {
        cudaStreamSynchronize(stream);
        e->ffn_hook(e->ffn_hook_data, layer, gate_buf, n_ff);
    }

    jinf_cuda_dequant_matvec(ffn_up, norm_out, up_buf, n_ff, n, qt_up, stream);

    jinf_cuda_silu_mul(gate_buf, gate_buf, up_buf, n_ff, stream);

    float* down_out = norm_out;
    jinf_cuda_dequant_matvec(ffn_down, gate_buf, down_out, n, n_ff, qt_down, stream);

    jinf_cuda_residual_add(hidden_state, down_out, n, stream);

    return JINF_OK;
}

// ---- Forward pass ----

jinf_status jinf_engine_forward(jinf_engine* e, const int32_t* tokens, int n_tokens,
                                 float** logits) {
    if (!e || !tokens || n_tokens <= 0 || !logits) return JINF_ERR_INVALID;

    jinf_timer timer;
    float* hidden_state = e->hidden_buf;

    // Process tokens one at a time for decode (MVP)
    for (int t = 0; t < n_tokens; t++) {
        // 1. Embed token
        embed_token(e, tokens[t], hidden_state);

        // 2. Process each layer
        for (int L = 0; L < e->n_layers; L++) {
            if (e->layers[L].is_hot) {
                jinf_status s = forward_layer_hot(e, L, hidden_state);
                if (s != JINF_OK) return s;
            } else {
                // Cold layer — NVMe double-buffer pipeline
                int next_cold = -1;
                for (int j = L + 1; j < e->n_layers; j++) {
                    if (!e->layers[j].is_hot) { next_cold = j; break; }
                }

                if (next_cold >= 0) {
                    uint64_t next_offset, next_size;
                    jinf_nvmw_get_layer_cold_range(e->model, next_cold,
                                                    &next_offset, &next_size);
                    jinf_buffer_start_read(e->buffers, e->io, e->fd_model,
                                            next_offset, (size_t)next_size);
                }

                void* cold_ptr = nullptr;
                size_t cold_size = 0;

                if (L == 0 || e->layers[L - 1].is_hot) {
                    uint64_t offset, size;
                    jinf_nvmw_get_layer_cold_range(e->model, L, &offset, &size);
                    void* buf = jinf_buffer_active_ptr(e->buffers);
                    jinf_io_read_sync(e->io, e->fd_model, buf, offset, size);
                    cold_ptr = buf;
                    cold_size = (size_t)size;
                } else {
                    jinf_buffer_wait_read(e->buffers, e->io, &cold_ptr, &cold_size);
                    jinf_buffer_swap(e->buffers);
                }

                jinf_status s = forward_layer_cold(e, L, hidden_state, cold_ptr);
                if (s != JINF_OK) return s;
            }
        }

        // 3. Final RMSNorm
        if (e->output_norm_weight) {
            jinf_cuda_rmsnorm(hidden_state, hidden_state, e->output_norm_weight,
                               e->n_embd, e->rms_norm_eps, e->output_norm_type,
                               e->compute_stream);
        }

        // 4. Output projection: logits = hidden_state @ output_weight
        if (e->output_weight) {
            jinf_cuda_dequant_matvec(e->output_weight, hidden_state, e->logits_buf,
                                      e->n_vocab, e->n_embd, e->output_weight_type,
                                      e->compute_stream);
        }

        e->n_past++;
    }

    cudaStreamSynchronize(e->compute_stream);

    *logits = e->logits_buf;
    e->perf.total_ms += timer.elapsed_ms();
    e->perf.tokens_generated += n_tokens;

    return JINF_OK;
}

jinf_status jinf_engine_generate(jinf_engine* e, const int32_t* prompt, int n_prompt,
                                  int32_t* output, int max_tokens, int* n_generated) {
    if (!e || !output || !n_generated) return JINF_ERR_INVALID;

    // Process prompt (or skip if n_prompt=0, using existing logits from prior forward)
    float* logits = nullptr;
    if (n_prompt > 0) {
        if (!prompt) return JINF_ERR_INVALID;
        jinf_status s = jinf_engine_forward(e, prompt, n_prompt, &logits);
        if (s != JINF_OK) return s;
    } else {
        if (e->n_past == 0) return JINF_ERR_INVALID;
        logits = e->logits_buf;
    }

    int generated = 0;

    for (int i = 0; i < max_tokens; i++) {
        // Greedy sampling: argmax of logits (uses pre-allocated pinned buffer)
        cudaMemcpy(e->host_logits, logits, e->n_vocab * sizeof(float), cudaMemcpyDeviceToHost);

        int best_id = 0;
        float best_val = e->host_logits[0];
        for (int j = 1; j < e->n_vocab; j++) {
            if (e->host_logits[j] > best_val) {
                best_val = e->host_logits[j];
                best_id = j;
            }
        }

        output[generated++] = best_id;

        // Check for EOS (token_id 2 is typical for Llama)
        if (best_id == 2) break;

        // Feed the generated token back
        jinf_status ds = jinf_engine_forward(e, &best_id, 1, &logits);
        if (ds != JINF_OK) break;
    }

    *n_generated = generated;
    return JINF_OK;
}
