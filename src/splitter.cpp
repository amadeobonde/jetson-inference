#include "jinf/splitter.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <string>

// ---- Helper: parse layer index from tensor name ----

static int parse_layer_index(const char* name) {
    // Match patterns like "blk.0.attn_q.weight" → layer 0
    const char* p = strstr(name, "blk.");
    if (!p) return -1;
    return atoi(p + 4);
}

// ---- Helper: classify tensor by role ----

enum tensor_role {
    ROLE_EMBED,        // token_embd
    ROLE_OUTPUT,       // output.weight
    ROLE_OUTPUT_NORM,  // output_norm.weight
    ROLE_ATTN_NORM,    // blk.N.attn_norm
    ROLE_FFN_NORM,     // blk.N.ffn_norm
    ROLE_ATTN_Q,
    ROLE_ATTN_K,
    ROLE_ATTN_V,
    ROLE_ATTN_OUTPUT,
    ROLE_FFN_GATE,
    ROLE_FFN_UP,
    ROLE_FFN_DOWN,
    ROLE_OTHER,
};

static tensor_role classify_tensor(const char* name) {
    if (strcmp(name, "token_embd.weight") == 0)  return ROLE_EMBED;
    if (strcmp(name, "output.weight") == 0)       return ROLE_OUTPUT;
    if (strcmp(name, "output_norm.weight") == 0)  return ROLE_OUTPUT_NORM;

    if (strstr(name, "attn_norm.weight")) return ROLE_ATTN_NORM;
    if (strstr(name, "ffn_norm.weight"))  return ROLE_FFN_NORM;
    if (strstr(name, "attn_q.weight"))    return ROLE_ATTN_Q;
    if (strstr(name, "attn_k.weight"))    return ROLE_ATTN_K;
    if (strstr(name, "attn_v.weight"))    return ROLE_ATTN_V;
    if (strstr(name, "attn_output.weight")) return ROLE_ATTN_OUTPUT;
    if (strstr(name, "ffn_gate.weight"))  return ROLE_FFN_GATE;
    if (strstr(name, "ffn_up.weight"))    return ROLE_FFN_UP;
    if (strstr(name, "ffn_down.weight"))  return ROLE_FFN_DOWN;

    return ROLE_OTHER;
}

static bool is_always_hot(tensor_role role) {
    return role == ROLE_EMBED || role == ROLE_OUTPUT || role == ROLE_OUTPUT_NORM ||
           role == ROLE_ATTN_NORM || role == ROLE_FFN_NORM;
}

// ---- Public API ----

jinf_status jinf_split_analyze(const jinf_gguf_file* model,
                                const jinf_split_config* config,
                                jinf_split_plan** plan_out) {
    if (!model || !config || !plan_out) return JINF_ERR_INVALID;

    int n = (int)model->tensor_count;

    jinf_split_plan* plan = (jinf_split_plan*)calloc(1, sizeof(jinf_split_plan));
    if (!plan) return JINF_ERR_OOM;

    plan->assignments = (jinf_weight_assignment*)calloc(n, sizeof(jinf_weight_assignment));
    if (!plan->assignments) { free(plan); return JINF_ERR_OOM; }
    plan->count = n;

    size_t hot_bytes = 0;
    size_t cold_bytes = 0;
    size_t budget = config->gpu_memory_budget;

    // Phase 1: assign always-hot tensors
    for (int i = 0; i < n; i++) {
        const jinf_gguf_tensor_info* t = &model->tensor_infos[i];
        tensor_role role = classify_tensor(t->name);
        int layer = parse_layer_index(t->name);

        plan->assignments[i].tensor = t;
        plan->assignments[i].layer_index = layer;

        if (is_always_hot(role)) {
            plan->assignments[i].placement = JINF_PLACE_HOT;
            hot_bytes += t->n_bytes;
        } else {
            plan->assignments[i].placement = JINF_PLACE_COLD;
            cold_bytes += t->n_bytes;
        }
    }

    // Phase 2: fill remaining budget by promoting layers (layer 0 first)
    // Collect cold tensors grouped by layer, sorted by layer index
    struct layer_group {
        int layer;
        std::vector<int> attn_indices;
        std::vector<int> ffn_indices;
        size_t attn_bytes;
        size_t ffn_bytes;
    };

    std::vector<layer_group> groups;

    for (int i = 0; i < n; i++) {
        if (plan->assignments[i].placement != JINF_PLACE_COLD) continue;
        int layer = plan->assignments[i].layer_index;
        if (layer < 0) continue;

        // Find or create group
        layer_group* g = nullptr;
        for (auto& grp : groups) {
            if (grp.layer == layer) { g = &grp; break; }
        }
        if (!g) {
            groups.push_back({layer, {}, {}, 0, 0});
            g = &groups.back();
        }

        tensor_role role = classify_tensor(model->tensor_infos[i].name);
        if (role >= ROLE_ATTN_Q && role <= ROLE_ATTN_OUTPUT) {
            g->attn_indices.push_back(i);
            g->attn_bytes += model->tensor_infos[i].n_bytes;
        } else {
            g->ffn_indices.push_back(i);
            g->ffn_bytes += model->tensor_infos[i].n_bytes;
        }
    }

    std::sort(groups.begin(), groups.end(),
              [](const layer_group& a, const layer_group& b) { return a.layer < b.layer; });

    // Promote attention first, then FFN, layer by layer
    for (auto& g : groups) {
        if (hot_bytes + g.attn_bytes <= budget) {
            for (int idx : g.attn_indices) {
                plan->assignments[idx].placement = JINF_PLACE_HOT;
                hot_bytes += model->tensor_infos[idx].n_bytes;
                cold_bytes -= model->tensor_infos[idx].n_bytes;
            }
        } else {
            break;  // no more budget
        }

        if (hot_bytes + g.ffn_bytes <= budget) {
            for (int idx : g.ffn_indices) {
                plan->assignments[idx].placement = JINF_PLACE_HOT;
                hot_bytes += model->tensor_infos[idx].n_bytes;
                cold_bytes -= model->tensor_infos[idx].n_bytes;
            }
        }
    }

    plan->total_hot_bytes = hot_bytes;
    plan->total_cold_bytes = cold_bytes;

    *plan_out = plan;
    return JINF_OK;
}

void jinf_split_plan_free(jinf_split_plan* plan) {
    if (!plan) return;
    free(plan->assignments);
    free(plan);
}

void jinf_split_plan_print(const jinf_split_plan* plan) {
    if (!plan) return;

    int n_hot = 0, n_cold = 0;
    for (int i = 0; i < plan->count; i++) {
        if (plan->assignments[i].placement == JINF_PLACE_HOT) n_hot++;
        else n_cold++;
    }

    fprintf(stderr, "[jinf] Split plan: %d tensors (%d hot, %d cold)\n",
            plan->count, n_hot, n_cold);
    fprintf(stderr, "[jinf]   Hot:  %.1f MB\n", plan->total_hot_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "[jinf]   Cold: %.1f MB\n", plan->total_cold_bytes / (1024.0 * 1024.0));

    for (int i = 0; i < plan->count; i++) {
        const jinf_weight_assignment* a = &plan->assignments[i];
        fprintf(stderr, "[jinf]   %-50s  %s  %8.2f MB  layer=%d\n",
                a->tensor->name,
                a->placement == JINF_PLACE_HOT ? "HOT " : "COLD",
                a->tensor->n_bytes / (1024.0 * 1024.0),
                a->layer_index);
    }
}
