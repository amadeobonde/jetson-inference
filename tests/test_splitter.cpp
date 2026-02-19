// test_splitter — Unit tests for hot/cold weight splitter.

#include "jinf/splitter.h"
#include "jinf/gguf.h"
#include "jinf/quant.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

// Build a fake GGUF file with known tensors for testing
static jinf_gguf_file* create_test_model(uint32_t n_layers) {
    jinf_gguf_file* gf = (jinf_gguf_file*)calloc(1, sizeof(jinf_gguf_file));

    gf->hparams.n_layers = n_layers;
    gf->hparams.n_embd = 4096;
    gf->hparams.n_ff = 11008;

    // Count tensors: embd + output + output_norm + per-layer (attn_norm, ffn_norm, attn*4, ffn*3)
    int n_tensors = 3 + n_layers * 9;
    gf->tensor_count = n_tensors;
    gf->tensor_infos = (jinf_gguf_tensor_info*)calloc(n_tensors, sizeof(jinf_gguf_tensor_info));

    int idx = 0;
    auto add_tensor = [&](const char* name, int64_t rows, int64_t cols, jinf_qtype type) {
        jinf_gguf_tensor_info* t = &gf->tensor_infos[idx++];
        t->name = strdup(name);
        t->n_dims = 2;
        t->shape[0] = rows;
        t->shape[1] = cols;
        t->type = type;
        t->n_elements = rows * cols;
        t->n_bytes = jinf_tensor_nbytes(type, t->n_elements);
    };

    // Global tensors
    add_tensor("token_embd.weight", 4096, 32000, JINF_TYPE_Q4_K);
    add_tensor("output.weight", 4096, 32000, JINF_TYPE_Q4_K);
    add_tensor("output_norm.weight", 4096, 1, JINF_TYPE_F32);

    // Per-layer tensors
    char name[128];
    for (uint32_t l = 0; l < n_layers; l++) {
        snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", l);
        add_tensor(name, 4096, 1, JINF_TYPE_F32);

        snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", l);
        add_tensor(name, 4096, 1, JINF_TYPE_F32);

        snprintf(name, sizeof(name), "blk.%u.attn_q.weight", l);
        add_tensor(name, 4096, 4096, JINF_TYPE_Q4_K);

        snprintf(name, sizeof(name), "blk.%u.attn_k.weight", l);
        add_tensor(name, 1024, 4096, JINF_TYPE_Q4_K);

        snprintf(name, sizeof(name), "blk.%u.attn_v.weight", l);
        add_tensor(name, 1024, 4096, JINF_TYPE_Q4_K);

        snprintf(name, sizeof(name), "blk.%u.attn_output.weight", l);
        add_tensor(name, 4096, 4096, JINF_TYPE_Q4_K);

        snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", l);
        add_tensor(name, 11008, 4096, JINF_TYPE_Q4_K);

        snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", l);
        add_tensor(name, 11008, 4096, JINF_TYPE_Q4_K);

        snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", l);
        add_tensor(name, 4096, 11008, JINF_TYPE_Q4_K);
    }

    return gf;
}

static void free_test_model(jinf_gguf_file* gf) {
    for (uint64_t i = 0; i < gf->tensor_count; i++) {
        free(gf->tensor_infos[i].name);
    }
    free(gf->tensor_infos);
    free(gf);
}

static int test_always_hot() {
    printf("  test_always_hot...\n");

    jinf_gguf_file* model = create_test_model(2);

    // Budget = 0 — only always-hot should be hot
    jinf_split_config cfg = { .gpu_memory_budget = 0, .kv_cache_budget = 0, .buffer_budget = 0 };
    jinf_split_plan* plan = nullptr;

    assert(jinf_split_analyze(model, &cfg, &plan) == JINF_OK);

    int n_hot = 0;
    for (int i = 0; i < plan->count; i++) {
        if (plan->assignments[i].placement == JINF_PLACE_HOT) {
            n_hot++;
            const char* name = plan->assignments[i].tensor->name;
            // Should be embeddings, output, norms
            assert(strstr(name, "embd") || strstr(name, "output") || strstr(name, "norm"));
        }
    }

    // embd + output + output_norm + 2 * (attn_norm + ffn_norm) = 3 + 4 = 7
    assert(n_hot == 7);

    jinf_split_plan_free(plan);
    free_test_model(model);
    printf("    PASSED\n");
    return 0;
}

static int test_large_budget() {
    printf("  test_large_budget...\n");

    jinf_gguf_file* model = create_test_model(2);

    // Huge budget — everything should be hot
    jinf_split_config cfg = {
        .gpu_memory_budget = 100ULL * 1024 * 1024 * 1024,
        .kv_cache_budget = 0,
        .buffer_budget = 0,
    };
    jinf_split_plan* plan = nullptr;

    assert(jinf_split_analyze(model, &cfg, &plan) == JINF_OK);

    for (int i = 0; i < plan->count; i++) {
        assert(plan->assignments[i].placement == JINF_PLACE_HOT);
    }
    assert(plan->total_cold_bytes == 0);

    jinf_split_plan_free(plan);
    free_test_model(model);
    printf("    PASSED\n");
    return 0;
}

static int test_layer_promotion_order() {
    printf("  test_layer_promotion_order...\n");

    jinf_gguf_file* model = create_test_model(4);

    // Budget enough for always-hot + layer 0 attention
    // Attention per layer: Q(4096*4096) + K(1024*4096) + V(1024*4096) + O(4096*4096)
    // Q4_K: n_elements/256 * 144 bytes
    size_t attn_q_bytes = (4096LL * 4096 / 256) * 144;
    size_t attn_k_bytes = (1024LL * 4096 / 256) * 144;
    size_t attn_v_bytes = (1024LL * 4096 / 256) * 144;
    size_t attn_o_bytes = (4096LL * 4096 / 256) * 144;
    size_t attn_total = attn_q_bytes + attn_k_bytes + attn_v_bytes + attn_o_bytes;

    // Get always-hot size
    jinf_split_config cfg0 = { .gpu_memory_budget = 0, .kv_cache_budget = 0, .buffer_budget = 0 };
    jinf_split_plan* plan0 = nullptr;
    jinf_split_analyze(model, &cfg0, &plan0);
    size_t always_hot = plan0->total_hot_bytes;
    jinf_split_plan_free(plan0);

    // Set budget to allow always-hot + layer 0 attention
    jinf_split_config cfg = {
        .gpu_memory_budget = always_hot + attn_total + 1024,
        .kv_cache_budget = 0,
        .buffer_budget = 0,
    };
    jinf_split_plan* plan = nullptr;
    assert(jinf_split_analyze(model, &cfg, &plan) == JINF_OK);

    // Layer 0 attention should be hot, layer 0 FFN and other layers should be cold
    for (int i = 0; i < plan->count; i++) {
        const jinf_weight_assignment* a = &plan->assignments[i];
        if (a->layer_index == 0 && strstr(a->tensor->name, "attn_") &&
            !strstr(a->tensor->name, "norm")) {
            assert(a->placement == JINF_PLACE_HOT);
        }
        if (a->layer_index == 1 && strstr(a->tensor->name, "attn_q")) {
            assert(a->placement == JINF_PLACE_COLD);
        }
    }

    jinf_split_plan_free(plan);
    free_test_model(model);
    printf("    PASSED\n");
    return 0;
}

int main() {
    printf("=== test_splitter ===\n");

    test_always_hot();
    test_large_budget();
    test_layer_promotion_order();

    printf("All splitter tests passed!\n");
    return 0;
}
