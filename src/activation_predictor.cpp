#include "jinf/activation_predictor.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <numeric>

// ---- Phase 2: Activation Predictor Implementation ----

struct jinf_activation_predictor {
    jinf_predictor_config config;

    // Shared predictor weights: 2-layer MLP
    // Layer 1: [n_embd + layer_embd_dim] -> hidden_dim
    // Layer 2: hidden_dim -> n_ff
    float* w1;       // [hidden_dim x (n_embd + layer_embd_dim)]
    float* b1;       // [hidden_dim]
    float* w2;       // [n_ff x hidden_dim]
    float* b2;       // [n_ff]
    float* layer_emb; // [n_layers x layer_embd_dim]

    int layer_embd_dim;

    // Online accuracy tracking
    struct layer_stats {
        int total_predictions;
        int correct_predictions;
    };
    layer_stats* stats;
};

static const int LAYER_EMBD_DIM = 32;

jinf_status jinf_predictor_create(jinf_activation_predictor** pred,
                                   const jinf_predictor_config* config) {
    if (!pred || !config) return JINF_ERR_INVALID;
    if (config->n_embd == 0 || config->n_ff == 0) return JINF_ERR_INVALID;

    auto* p = new (std::nothrow) jinf_activation_predictor();
    if (!p) return JINF_ERR_OOM;

    p->config = *config;
    p->layer_embd_dim = LAYER_EMBD_DIM;

    int input_dim = config->n_embd + LAYER_EMBD_DIM;
    int hidden = config->hidden_dim;
    int output = config->n_ff;

    p->w1 = (float*)calloc(hidden * input_dim, sizeof(float));
    p->b1 = (float*)calloc(hidden, sizeof(float));
    p->w2 = (float*)calloc(output * hidden, sizeof(float));
    p->b2 = (float*)calloc(output, sizeof(float));
    p->layer_emb = (float*)calloc(config->n_layers * LAYER_EMBD_DIM, sizeof(float));

    if (!p->w1 || !p->b1 || !p->w2 || !p->b2 || !p->layer_emb) {
        jinf_predictor_destroy(p);
        return JINF_ERR_OOM;
    }

    p->stats = (jinf_activation_predictor::layer_stats*)calloc(
        config->n_layers, sizeof(jinf_activation_predictor::layer_stats));

    *pred = p;
    return JINF_OK;
}

void jinf_predictor_destroy(jinf_activation_predictor* pred) {
    if (!pred) return;
    free(pred->w1);
    free(pred->b1);
    free(pred->w2);
    free(pred->b2);
    free(pred->layer_emb);
    free(pred->stats);
    delete pred;
}

jinf_status jinf_predictor_load(jinf_activation_predictor* pred, const char* path) {
    if (!pred || !path) return JINF_ERR_INVALID;

    FILE* fp = fopen(path, "rb");
    if (!fp) return JINF_ERR_IO;

    int input_dim = pred->config.n_embd + pred->layer_embd_dim;
    int hidden = pred->config.hidden_dim;
    int output = pred->config.n_ff;
    int n_layers = pred->config.n_layers;

    size_t expected = 0;
    expected += hidden * input_dim * sizeof(float);
    expected += hidden * sizeof(float);
    expected += output * hidden * sizeof(float);
    expected += output * sizeof(float);
    expected += n_layers * LAYER_EMBD_DIM * sizeof(float);

    if (fread(pred->w1, sizeof(float), hidden * input_dim, fp) != (size_t)(hidden * input_dim)) goto fail;
    if (fread(pred->b1, sizeof(float), hidden, fp) != (size_t)hidden) goto fail;
    if (fread(pred->w2, sizeof(float), output * hidden, fp) != (size_t)(output * hidden)) goto fail;
    if (fread(pred->b2, sizeof(float), output, fp) != (size_t)output) goto fail;
    if (fread(pred->layer_emb, sizeof(float), n_layers * LAYER_EMBD_DIM, fp) !=
        (size_t)(n_layers * LAYER_EMBD_DIM)) goto fail;

    fclose(fp);
    return JINF_OK;

fail:
    fclose(fp);
    return JINF_ERR_PARSE;
}

jinf_status jinf_predictor_save(const jinf_activation_predictor* pred, const char* path) {
    if (!pred || !path) return JINF_ERR_INVALID;

    FILE* fp = fopen(path, "wb");
    if (!fp) return JINF_ERR_IO;

    int input_dim = pred->config.n_embd + pred->layer_embd_dim;
    int hidden = pred->config.hidden_dim;
    int output = pred->config.n_ff;
    int n_layers = pred->config.n_layers;

    fwrite(pred->w1, sizeof(float), hidden * input_dim, fp);
    fwrite(pred->b1, sizeof(float), hidden, fp);
    fwrite(pred->w2, sizeof(float), output * hidden, fp);
    fwrite(pred->b2, sizeof(float), output, fp);
    fwrite(pred->layer_emb, sizeof(float), n_layers * LAYER_EMBD_DIM, fp);

    fclose(fp);
    return JINF_OK;
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

jinf_status jinf_predictor_predict(jinf_activation_predictor* pred,
                                    const float* hidden_state,
                                    int layer_id,
                                    std::vector<int>& active_neurons) {
    if (!pred || !hidden_state) return JINF_ERR_INVALID;
    if (layer_id < 0 || layer_id >= pred->config.n_layers) return JINF_ERR_INVALID;

    int n_embd = pred->config.n_embd;
    int hidden = pred->config.hidden_dim;
    int n_ff = pred->config.n_ff;
    int input_dim = n_embd + pred->layer_embd_dim;
    float threshold = pred->config.threshold;

    // Construct input: [hidden_state; layer_embedding]
    std::vector<float> input(input_dim);
    memcpy(input.data(), hidden_state, n_embd * sizeof(float));
    memcpy(input.data() + n_embd,
           pred->layer_emb + layer_id * pred->layer_embd_dim,
           pred->layer_embd_dim * sizeof(float));

    // Layer 1: h = ReLU(w1 @ input + b1)
    std::vector<float> h(hidden);
    for (int i = 0; i < hidden; i++) {
        float sum = pred->b1[i];
        const float* row = pred->w1 + i * input_dim;
        for (int j = 0; j < input_dim; j++) {
            sum += row[j] * input[j];
        }
        h[i] = sum > 0 ? sum : 0;  // ReLU
    }

    // Layer 2: out = sigmoid(w2 @ h + b2)
    active_neurons.clear();
    for (int i = 0; i < n_ff; i++) {
        float sum = pred->b2[i];
        const float* row = pred->w2 + i * hidden;
        for (int j = 0; j < hidden; j++) {
            sum += row[j] * h[j];
        }
        if (sigmoid(sum) > threshold) {
            active_neurons.push_back(i);
        }
    }

    return JINF_OK;
}

void jinf_predictor_update_stats(jinf_activation_predictor* pred,
                                  int layer_id,
                                  const std::vector<int>& predicted,
                                  const std::vector<int>& actual) {
    if (!pred || !pred->stats) return;
    if (layer_id < 0 || layer_id >= pred->config.n_layers) return;

    // Compute intersection
    std::vector<int> p_sorted = predicted;
    std::vector<int> a_sorted = actual;
    std::sort(p_sorted.begin(), p_sorted.end());
    std::sort(a_sorted.begin(), a_sorted.end());

    std::vector<int> intersection;
    std::set_intersection(p_sorted.begin(), p_sorted.end(),
                          a_sorted.begin(), a_sorted.end(),
                          std::back_inserter(intersection));

    pred->stats[layer_id].total_predictions += (int)actual.size();
    pred->stats[layer_id].correct_predictions += (int)intersection.size();
}

float jinf_predictor_accuracy(const jinf_activation_predictor* pred, int layer_id) {
    if (!pred || !pred->stats) return 0.0f;
    if (layer_id < 0 || layer_id >= pred->config.n_layers) return 0.0f;

    int total = pred->stats[layer_id].total_predictions;
    if (total == 0) return 0.0f;
    return (float)pred->stats[layer_id].correct_predictions / (float)total;
}
