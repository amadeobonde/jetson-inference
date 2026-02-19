// inference_bench — End-to-end inference benchmark.
// Usage: ./inference_bench <model.nvmw> [--tokens N] [--context N]

#include "jinf/engine.h"
#include "jinf/common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct bench_args {
    const char* model_path;
    int max_tokens;
    int max_context;
    size_t gpu_budget_mb;
    size_t buffer_mb;
};

static bool parse_args(int argc, char** argv, bench_args* args) {
    args->model_path = nullptr;
    args->max_tokens = 32;
    args->max_context = 2048;
    args->gpu_budget_mb = 4500;
    args->buffer_mb = 512;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            args->max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--context") == 0 && i + 1 < argc) {
            args->max_context = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gpu-budget") == 0 && i + 1 < argc) {
            args->gpu_budget_mb = (size_t)atol(argv[++i]);
        } else if (strcmp(argv[i], "--buffer") == 0 && i + 1 < argc) {
            args->buffer_mb = (size_t)atol(argv[++i]);
        } else if (argv[i][0] != '-') {
            args->model_path = argv[i];
        }
    }

    if (!args->model_path) {
        fprintf(stderr, "Usage: %s <model.nvmw> [--tokens N] [--context N] "
                "[--gpu-budget MB] [--buffer MB]\n", argv[0]);
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    bench_args args;
    if (!parse_args(argc, argv, &args)) return 1;

    jinf_engine_config config = jinf_engine_config_default();
    config.model_path = args.model_path;
    config.gpu_memory_budget = args.gpu_budget_mb * 1024 * 1024;
    config.buffer_capacity = args.buffer_mb * 1024 * 1024;
    config.max_context = args.max_context;

    printf("=== Inference Benchmark ===\n");
    printf("Model:       %s\n", args.model_path);
    printf("Max tokens:  %d\n", args.max_tokens);
    printf("Max context: %d\n", args.max_context);
    printf("GPU budget:  %zu MB\n", args.gpu_budget_mb);
    printf("Buffer:      %zu MB\n\n", args.buffer_mb);

    jinf_engine* engine = nullptr;
    jinf_status s = jinf_engine_create(&engine, &config);
    if (s != JINF_OK) {
        fprintf(stderr, "Failed to create engine: %s\n", jinf_status_str(s));
        return 1;
    }

    // Hardcoded prompt tokens (BOS + common tokens)
    // In practice, you'd use a tokenizer. For benchmarking, any tokens work.
    std::vector<int32_t> prompt = {1, 450, 4696, 310, 2834, 338};  // "The capital of France is"
    int prompt_len = (int)prompt.size();

    // --- Prefill benchmark ---
    printf("--- Prefill (%d tokens) ---\n", prompt_len);
    jinf_timer prefill_timer;

    float* logits = nullptr;
    s = jinf_engine_forward(engine, prompt.data(), prompt_len, &logits);
    if (s != JINF_OK) {
        fprintf(stderr, "Prefill failed: %s\n", jinf_status_str(s));
        jinf_engine_destroy(engine);
        return 1;
    }

    double prefill_ms = prefill_timer.elapsed_ms();
    printf("  Time:       %.1f ms\n", prefill_ms);
    printf("  Throughput: %.1f tokens/sec\n", prompt_len / (prefill_ms / 1000.0));

    // --- Decode benchmark ---
    printf("\n--- Decode (%d tokens) ---\n", args.max_tokens);
    std::vector<int32_t> output(args.max_tokens);
    int n_generated = 0;

    jinf_timer decode_timer;

    s = jinf_engine_generate(engine, prompt.data(), 0, output.data(),
                              args.max_tokens, &n_generated);

    double decode_ms = decode_timer.elapsed_ms();

    if (s == JINF_OK && n_generated > 0) {
        printf("  Generated:  %d tokens\n", n_generated);
        printf("  Time:       %.1f ms\n", decode_ms);
        printf("  Throughput: %.2f tokens/sec\n", n_generated / (decode_ms / 1000.0));
        printf("  Per token:  %.1f ms\n", decode_ms / n_generated);
    } else {
        printf("  Generation failed or produced 0 tokens\n");
    }

    // --- Stats ---
    const jinf_perf_stats* stats = jinf_engine_get_stats(engine);
    if (stats) {
        printf("\n--- Performance Stats ---\n");
        printf("  Total time:     %.1f ms\n", stats->total_ms);
        printf("  NVMe reads:     %.1f ms\n", stats->nvme_read_ms);
        printf("  GPU compute:    %.1f ms\n", stats->gpu_compute_ms);
        printf("  NVMe bandwidth: %.1f MB read\n", stats->nvme_bytes_read / (1024.0 * 1024.0));
    }

    // --- Memory breakdown ---
    printf("\n--- Memory ---\n");
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("  GPU total: %zu MB\n", total_mem / (1024 * 1024));
    printf("  GPU free:  %zu MB\n", free_mem / (1024 * 1024));
    printf("  GPU used:  %zu MB\n", (total_mem - free_mem) / (1024 * 1024));

    // Output token IDs
    printf("\n--- Generated tokens ---\n  ");
    for (int i = 0; i < n_generated; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");

    jinf_engine_destroy(engine);
    return 0;
}
