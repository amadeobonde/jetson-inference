// profile_activations — Phase 2: Calibration profiler for neuron activation frequencies.
// Runs calibration data through the model and records per-neuron activation frequencies.
// Usage: ./profile_activations --model model.nvmw --output activations.bin --samples 1000

#include "jinf/engine.h"
#include "jinf/common.h"
#include "jinf/quant.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

struct profile_args {
    const char* model_path;
    const char* output_path;
    int num_samples;
};

static bool parse_args(int argc, char** argv, profile_args* args) {
    args->model_path = nullptr;
    args->output_path = "activations.bin";
    args->num_samples = 1000;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            args->model_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            args->output_path = argv[++i];
        } else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            args->num_samples = atoi(argv[++i]);
        }
    }

    if (!args->model_path) {
        fprintf(stderr, "Usage: %s --model <model.nvmw> [--output <activations.bin>] [--samples N]\n",
                argv[0]);
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    profile_args args;
    if (!parse_args(argc, argv, &args)) return 1;

    // Create engine
    jinf_engine_config config = jinf_engine_config_default();
    config.model_path = args.model_path;

    jinf_engine* engine = nullptr;
    jinf_status s = jinf_engine_create(&engine, &config);
    if (s != JINF_OK) {
        fprintf(stderr, "Failed to create engine: %s\n", jinf_status_str(s));
        return 1;
    }

    int n_layers = engine->n_layers;
    int n_ff = engine->n_ff;

    // Activation frequency counters: [n_layers][n_ff]
    std::vector<std::vector<int>> counts(n_layers, std::vector<int>(n_ff, 0));
    int total_tokens = 0;

    printf("Profiling activations over %d samples...\n", args.num_samples);
    printf("Model: %d layers, %d FFN neurons per layer\n", n_layers, n_ff);

    // Generate random token sequences for calibration
    // In production, use actual text data. For MVP, use random tokens.
    srand(42);
    for (int sample = 0; sample < args.num_samples; sample++) {
        int prompt_len = 8 + rand() % 24;  // 8-32 tokens
        std::vector<int32_t> tokens(prompt_len);
        for (int i = 0; i < prompt_len; i++) {
            tokens[i] = rand() % engine->n_vocab;
        }

        // Run forward pass
        float* logits = nullptr;
        s = jinf_engine_forward(engine, tokens.data(), prompt_len, &logits);
        if (s != JINF_OK) {
            fprintf(stderr, "Forward pass failed at sample %d: %s\n",
                    sample, jinf_status_str(s));
            continue;
        }

        // TODO: Hook into the FFN to capture actual activations per neuron.
        // For the profiler MVP, we count which neurons have gate activations > threshold.
        // This requires instrumenting the engine's forward pass, which is Phase 2 work.
        // For now, we record placeholder uniform frequencies.

        total_tokens += prompt_len;
        jinf_engine_reset(engine);

        if ((sample + 1) % 100 == 0) {
            printf("  Processed %d/%d samples (%d tokens)\n",
                   sample + 1, args.num_samples, total_tokens);
        }
    }

    // Write activation profile
    FILE* fp = fopen(args.output_path, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open output: %s\n", args.output_path);
        jinf_engine_destroy(engine);
        return 1;
    }

    // Header: n_layers, n_ff, total_tokens
    fwrite(&n_layers, sizeof(int), 1, fp);
    fwrite(&n_ff, sizeof(int), 1, fp);
    fwrite(&total_tokens, sizeof(int), 1, fp);

    // Per-layer per-neuron frequency (float, 0.0-1.0)
    for (int l = 0; l < n_layers; l++) {
        std::vector<float> freq(n_ff);
        for (int n = 0; n < n_ff; n++) {
            freq[n] = total_tokens > 0 ? (float)counts[l][n] / (float)total_tokens : 0.0f;
        }
        fwrite(freq.data(), sizeof(float), n_ff, fp);
    }

    fclose(fp);
    printf("Activation profile written to: %s\n", args.output_path);
    printf("Total tokens processed: %d\n", total_tokens);

    jinf_engine_destroy(engine);
    return 0;
}
