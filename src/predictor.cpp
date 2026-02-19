#include "jinf/predictor.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct jinf_activation_predictor {
    int n_layers;
    int n_ff;
    int total_tokens;
    float threshold;

    // Per-layer precomputed active neuron lists (static predictor)
    // active_neurons[layer] = sorted list of neuron IDs that exceed threshold
    std::vector<std::vector<int>> active_neurons;

    // Raw frequencies for save/inspection
    std::vector<std::vector<float>> frequencies;
};

jinf_status jinf_predictor_create(jinf_activation_predictor** pred,
                                   const char* profile_path,
                                   float threshold) {
    if (!pred || !profile_path) return JINF_ERR_INVALID;

    FILE* fp = fopen(profile_path, "rb");
    if (!fp) {
        JINF_LOG("Failed to open activation profile: %s", profile_path);
        return JINF_ERR_IO;
    }

    int n_layers, n_ff, total_tokens;
    if (fread(&n_layers, sizeof(int), 1, fp) != 1 ||
        fread(&n_ff, sizeof(int), 1, fp) != 1 ||
        fread(&total_tokens, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        return JINF_ERR_PARSE;
    }

    if (n_layers <= 0 || n_ff <= 0) {
        fclose(fp);
        return JINF_ERR_PARSE;
    }

    jinf_activation_predictor* p = new (std::nothrow) jinf_activation_predictor();
    if (!p) { fclose(fp); return JINF_ERR_OOM; }

    p->n_layers = n_layers;
    p->n_ff = n_ff;
    p->total_tokens = total_tokens;
    p->threshold = threshold;
    p->frequencies.resize(n_layers);
    p->active_neurons.resize(n_layers);

    for (int l = 0; l < n_layers; l++) {
        p->frequencies[l].resize(n_ff);
        if (fread(p->frequencies[l].data(), sizeof(float), n_ff, fp) != (size_t)n_ff) {
            delete p;
            fclose(fp);
            return JINF_ERR_PARSE;
        }

        // Build active neuron list for this layer
        for (int i = 0; i < n_ff; i++) {
            if (p->frequencies[l][i] > threshold) {
                p->active_neurons[l].push_back(i);
            }
        }
    }

    fclose(fp);

    // Log statistics
    int total_active = 0;
    for (int l = 0; l < n_layers; l++) {
        total_active += (int)p->active_neurons[l].size();
    }
    float avg_active = (float)total_active / n_layers;
    float avg_pct = 100.0f * avg_active / n_ff;

    JINF_LOG("Predictor: %d layers, %d neurons, threshold=%.2f", n_layers, n_ff, threshold);
    JINF_LOG("Predictor: avg %.0f/%.0f neurons active (%.1f%%)", avg_active, (float)n_ff, avg_pct);

    *pred = p;
    return JINF_OK;
}

void jinf_predictor_destroy(jinf_activation_predictor* pred) {
    delete pred;
}

jinf_status jinf_predictor_predict(const jinf_activation_predictor* pred,
                                    const float* /*hidden_state*/,
                                    int layer,
                                    int* active_ids,
                                    int* n_active) {
    if (!pred || !active_ids || !n_active) return JINF_ERR_INVALID;
    if (layer < 0 || layer >= pred->n_layers) return JINF_ERR_INVALID;

    const auto& actives = pred->active_neurons[layer];
    int count = (int)actives.size();

    memcpy(active_ids, actives.data(), count * sizeof(int));
    *n_active = count;

    return JINF_OK;
}

int jinf_predictor_n_layers(const jinf_activation_predictor* pred) {
    return pred ? pred->n_layers : 0;
}

int jinf_predictor_n_ff(const jinf_activation_predictor* pred) {
    return pred ? pred->n_ff : 0;
}

jinf_status jinf_predictor_save(const jinf_activation_predictor* pred, const char* path) {
    if (!pred || !path) return JINF_ERR_INVALID;

    FILE* fp = fopen(path, "wb");
    if (!fp) return JINF_ERR_IO;

    fwrite(&pred->n_layers, sizeof(int), 1, fp);
    fwrite(&pred->n_ff, sizeof(int), 1, fp);
    fwrite(&pred->total_tokens, sizeof(int), 1, fp);

    for (int l = 0; l < pred->n_layers; l++) {
        fwrite(pred->frequencies[l].data(), sizeof(float), pred->n_ff, fp);
    }

    fclose(fp);
    return JINF_OK;
}

jinf_status jinf_predictor_load(jinf_activation_predictor* pred, const char* path) {
    // For static predictor, load is the same as create.
    // This is a no-op stub for the MLP predictor future.
    (void)pred;
    (void)path;
    return JINF_OK;
}
