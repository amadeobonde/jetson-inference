// test_activation_predictor — Phase 2: Unit tests for activation predictor.

#include "jinf/activation_predictor.h"
#include "jinf/common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <vector>

static int test_predictor_create() {
    printf("  test_predictor_create...\n");

    jinf_predictor_config cfg = jinf_predictor_config_default();
    cfg.n_embd = 128;
    cfg.n_ff = 256;
    cfg.n_layers = 4;
    cfg.hidden_dim = 64;

    jinf_activation_predictor* pred = nullptr;
    assert(jinf_predictor_create(&pred, &cfg) == JINF_OK);
    assert(pred != nullptr);

    jinf_predictor_destroy(pred);
    printf("    PASSED\n");
    return 0;
}

static int test_predictor_predict() {
    printf("  test_predictor_predict...\n");

    jinf_predictor_config cfg = jinf_predictor_config_default();
    cfg.n_embd = 64;
    cfg.n_ff = 128;
    cfg.n_layers = 2;
    cfg.hidden_dim = 32;
    cfg.threshold = 0.5f;

    jinf_activation_predictor* pred = nullptr;
    assert(jinf_predictor_create(&pred, &cfg) == JINF_OK);

    // With zero weights, sigmoid(0) = 0.5. With threshold 0.5, nothing should activate.
    // But biases are also 0, so sigmoid(b2[i]) = 0.5. Exactly at threshold.
    // Set threshold slightly below to test.
    // Actually, with all-zero weights, output = sigmoid(0) = 0.5 for all neurons.

    float hidden[64];
    for (int i = 0; i < 64; i++) hidden[i] = 0.0f;

    std::vector<int> active;
    assert(jinf_predictor_predict(pred, hidden, 0, active) == JINF_OK);
    // With all-zero weights and bias, sigmoid(0) = 0.5, and threshold = 0.5
    // The comparison is >, so no neurons should be above 0.5
    assert(active.empty());

    jinf_predictor_destroy(pred);
    printf("    PASSED\n");
    return 0;
}

static int test_predictor_save_load() {
    printf("  test_predictor_save_load...\n");

    jinf_predictor_config cfg = jinf_predictor_config_default();
    cfg.n_embd = 32;
    cfg.n_ff = 64;
    cfg.n_layers = 2;
    cfg.hidden_dim = 16;

    jinf_activation_predictor* pred = nullptr;
    assert(jinf_predictor_create(&pred, &cfg) == JINF_OK);

    const char* path = "/tmp/test_jinf_predictor.bin";
    assert(jinf_predictor_save(pred, path) == JINF_OK);

    jinf_activation_predictor* loaded = nullptr;
    assert(jinf_predictor_create(&loaded, &cfg) == JINF_OK);
    assert(jinf_predictor_load(loaded, path) == JINF_OK);

    // Both should produce same results
    float hidden[32] = {0};
    std::vector<int> active1, active2;
    jinf_predictor_predict(pred, hidden, 0, active1);
    jinf_predictor_predict(loaded, hidden, 0, active2);
    assert(active1.size() == active2.size());

    jinf_predictor_destroy(pred);
    jinf_predictor_destroy(loaded);
    printf("    PASSED\n");
    return 0;
}

static int test_predictor_accuracy() {
    printf("  test_predictor_accuracy...\n");

    jinf_predictor_config cfg = jinf_predictor_config_default();
    cfg.n_embd = 32;
    cfg.n_ff = 64;
    cfg.n_layers = 2;
    cfg.hidden_dim = 16;

    jinf_activation_predictor* pred = nullptr;
    assert(jinf_predictor_create(&pred, &cfg) == JINF_OK);

    // Initially accuracy should be 0
    assert(jinf_predictor_accuracy(pred, 0) == 0.0f);

    // Update with perfect predictions
    std::vector<int> predicted = {1, 2, 3};
    std::vector<int> actual = {1, 2, 3};
    jinf_predictor_update_stats(pred, 0, predicted, actual);
    assert(jinf_predictor_accuracy(pred, 0) == 1.0f);

    // Update with half-correct predictions
    std::vector<int> predicted2 = {10, 20, 30};
    std::vector<int> actual2 = {10, 20, 40};
    jinf_predictor_update_stats(pred, 0, predicted2, actual2);
    // Total: 3 + 3 = 6 actual, 3 + 2 = 5 correct
    float expected_accuracy = 5.0f / 6.0f;
    float actual_accuracy = jinf_predictor_accuracy(pred, 0);
    assert(fabsf(actual_accuracy - expected_accuracy) < 0.01f);

    jinf_predictor_destroy(pred);
    printf("    PASSED\n");
    return 0;
}

int main() {
    printf("=== test_activation_predictor (Phase 2) ===\n");

    test_predictor_create();
    test_predictor_predict();
    test_predictor_save_load();
    test_predictor_accuracy();

    printf("All activation predictor tests passed!\n");
    return 0;
}
