#pragma once

#include "jinf/common.h"
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

// ---- Phase 2: Activation Predictor ----
//
// Lightweight MLP that predicts which FFN neurons will activate for a given
// hidden state, enabling sparse NVMe reads of only the needed neuron bundles.

struct jinf_predictor_config {
    int   n_embd;          // model embedding dimension
    int   n_ff;            // FFN intermediate size
    int   hidden_dim;      // predictor hidden layer size (default 512)
    int   n_layers;        // number of model layers
    float threshold;       // activation prediction threshold (default 0.5)
    bool  use_shared;      // use one shared predictor with layer embedding
};

static inline jinf_predictor_config jinf_predictor_config_default() {
    return {
        .n_embd     = 0,
        .n_ff       = 0,
        .hidden_dim = 512,
        .n_layers   = 0,
        .threshold  = 0.5f,
        .use_shared = true,
    };
}

struct jinf_activation_predictor;

// ---- API ----

jinf_status jinf_predictor_create(jinf_activation_predictor** pred,
                                   const jinf_predictor_config* config);
void        jinf_predictor_destroy(jinf_activation_predictor* pred);

// Load trained predictor weights from file.
jinf_status jinf_predictor_load(jinf_activation_predictor* pred, const char* path);

// Save predictor weights to file.
jinf_status jinf_predictor_save(const jinf_activation_predictor* pred, const char* path);

// Predict active neuron indices for a given layer.
// Returns sorted vector of neuron IDs predicted to activate.
jinf_status jinf_predictor_predict(jinf_activation_predictor* pred,
                                    const float* hidden_state,
                                    int layer_id,
                                    std::vector<int>& active_neurons);

// Update online accuracy tracking (for monitoring prediction quality).
void jinf_predictor_update_stats(jinf_activation_predictor* pred,
                                  int layer_id,
                                  const std::vector<int>& predicted,
                                  const std::vector<int>& actual);

// Get prediction accuracy for a layer (0.0 to 1.0).
float jinf_predictor_accuracy(const jinf_activation_predictor* pred, int layer_id);

// ---- Static predictor API (Phase 2 MVP) ----
// Creates a predictor from an activation profile file (frequencies).
// Neurons with frequency > threshold are always included.
jinf_status jinf_predictor_create_static(jinf_activation_predictor** pred,
                                          const char* profile_path,
                                          float threshold);

// Predict with plain C arrays (no std::vector dependency).
// active_ids: caller-provided buffer [n_ff max], n_active: output count.
jinf_status jinf_predictor_predict_static(const jinf_activation_predictor* pred,
                                           const float* hidden_state,
                                           int layer_id,
                                           int* active_ids,
                                           int* n_active);

// Accessors for model dimensions.
int jinf_predictor_n_layers(const jinf_activation_predictor* pred);
int jinf_predictor_n_ff(const jinf_activation_predictor* pred);
