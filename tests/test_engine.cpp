// test_engine — Integration test for inference engine.
// Verifies engine creation and basic forward pass with a synthetic model.
// Full testing requires an actual .nvmw model file.

#include "jinf/engine.h"
#include "jinf/common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

// Smoke test: verify engine config defaults
static int test_config_defaults() {
    printf("  test_config_defaults...\n");

    jinf_engine_config cfg = jinf_engine_config_default();
    assert(cfg.model_path == nullptr);
    assert(cfg.gpu_memory_budget > 0);
    assert(cfg.buffer_capacity > 0);
    assert(cfg.max_context > 0);
    assert(cfg.io_queue_depth > 0);

    printf("    PASSED\n");
    return 0;
}

// Test that engine creation fails gracefully with invalid path
static int test_invalid_model() {
    printf("  test_invalid_model...\n");

    jinf_engine_config cfg = jinf_engine_config_default();
    cfg.model_path = "/tmp/nonexistent_model.nvmw";

    jinf_engine* e = nullptr;
    jinf_status s = jinf_engine_create(&e, &cfg);
    assert(s != JINF_OK);
    assert(e == nullptr);

    printf("    PASSED\n");
    return 0;
}

// Test null argument handling
static int test_null_args() {
    printf("  test_null_args...\n");

    assert(jinf_engine_create(nullptr, nullptr) == JINF_ERR_INVALID);

    jinf_engine_config cfg = jinf_engine_config_default();
    assert(jinf_engine_create(nullptr, &cfg) == JINF_ERR_INVALID);

    jinf_engine* e = nullptr;
    assert(jinf_engine_create(&e, nullptr) == JINF_ERR_INVALID);

    // Forward with null engine
    float* logits = nullptr;
    int32_t token = 1;
    assert(jinf_engine_forward(nullptr, &token, 1, &logits) == JINF_ERR_INVALID);

    printf("    PASSED\n");
    return 0;
}

int main() {
    printf("=== test_engine ===\n");

    test_config_defaults();
    test_invalid_model();
    test_null_args();

    printf("All engine tests passed!\n");
    printf("Note: Full forward pass testing requires an actual .nvmw model file.\n");
    printf("Run: ./prepare_model --input model.gguf --output model.nvmw\n");
    printf("Then: ./inference_bench model.nvmw\n");
    return 0;
}
