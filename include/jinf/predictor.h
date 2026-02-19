#pragma once

#include "jinf/common.h"
#include <cstdint>
#include <cstddef>

// ---- Phase 2: Activation Predictor ----
//
// Predicts which FFN neurons will activate for a given hidden state.
// MVP: "static predictor" mode — uses profiled per-neuron frequencies as thresholds.
// If neuron (layer, i) activated more than `threshold` fraction of the time during
// profiling, it is always included. Otherwise it is excluded.
//
// Future: learned MLP predictor (2-layer, hidden_size=512, per-layer).

// Activation profile file format:
//   int32 n_layers
//   int32 n_ff
//   int32 total_tokens
//   float frequencies[n_layers][n_ff]   // activation rate 0.0-1.0

struct jinf_activation_predictor;

// Create a static predictor from an activation profile file.
// threshold: neurons with frequency > threshold are always included (e.g., 0.5).
jinf_status jinf_predictor_create(jinf_activation_predictor** pred,
                                   const char* profile_path,
                                   float threshold);

void jinf_predictor_destroy(jinf_activation_predictor* pred);

// Predict which neurons should be activated for a given layer.
// For the static predictor, the input hidden_state is ignored —
// the prediction is based solely on profiled frequencies.
// Returns the list of active neuron IDs and count.
// active_ids: caller-provided buffer [n_ff max]
// n_active: output count
jinf_status jinf_predictor_predict(const jinf_activation_predictor* pred,
                                    const float* hidden_state,  // unused for static predictor
                                    int layer,
                                    int* active_ids,
                                    int* n_active);

// Get model dimensions from the predictor.
int jinf_predictor_n_layers(const jinf_activation_predictor* pred);
int jinf_predictor_n_ff(const jinf_activation_predictor* pred);

// Save/load predictor weights (for MLP predictor, future).
jinf_status jinf_predictor_save(const jinf_activation_predictor* pred, const char* path);
jinf_status jinf_predictor_load(jinf_activation_predictor* pred, const char* path);
