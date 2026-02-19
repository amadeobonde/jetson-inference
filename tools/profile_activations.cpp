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
    float threshold;
};

static bool parse_args(int argc, char** argv, profile_args* args) {
    args->model_path = nullptr;
    args->output_path = "activations.bin";
    args->num_samples = 1000;
    args->threshold = 0.1f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            args->model_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            args->output_path = argv[++i];
        } else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            args->num_samples = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            args->threshold = (float)atof(argv[++i]);
        }
    }

    if (!args->model_path) {
        fprintf(stderr, "Usage: %s --model <model.nvmw> [--output <activations.bin>] "
                "[--samples N] [--threshold F]\n", argv[0]);
        return false;
    }
    return true;
}

// FFN hook callback data
struct hook_state {
    std::vector<std::vector<int>>* counts;
    std::vector<float>* host_buf;   // reusable host buffer [n_ff]
    float threshold;
    int n_ff;
};

// FFN profiling callback: copies gate output to host, applies SiLU threshold
static void ffn_profile_hook(void* user_data, int layer, const float* gate_output, int n_ff) {
    hook_state* hs = (hook_state*)user_data;

    // Copy gate values from GPU to host
    cudaMemcpy(hs->host_buf->data(), gate_output, n_ff * sizeof(float), cudaMemcpyDeviceToHost);

    // Check |SiLU(gate[i])| > threshold
    for (int i = 0; i < n_ff; i++) {
        float g = (*hs->host_buf)[i];
        float silu_val = g / (1.0f + expf(-g));
        if (fabsf(silu_val) > hs->threshold) {
            (*hs->counts)[layer][i]++;
        }
    }
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
    std::vector<float> host_buf(n_ff);
    int total_tokens = 0;

    // Set up FFN profiling hook
    hook_state hs;
    hs.counts = &counts;
    hs.host_buf = &host_buf;
    hs.threshold = args.threshold;
    hs.n_ff = n_ff;

    engine->ffn_hook = ffn_profile_hook;
    engine->ffn_hook_data = &hs;

    printf("Profiling activations over %d samples (threshold=%.2f)...\n",
           args.num_samples, args.threshold);
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

        // Run forward pass — hook captures activations automatically
        float* logits = nullptr;
        s = jinf_engine_forward(engine, tokens.data(), prompt_len, &logits);
        if (s != JINF_OK) {
            fprintf(stderr, "Forward pass failed at sample %d: %s\n",
                    sample, jinf_status_str(s));
            continue;
        }

        total_tokens += prompt_len;
        jinf_engine_reset(engine);

        if ((sample + 1) % 100 == 0) {
            printf("  Processed %d/%d samples (%d tokens)\n",
                   sample + 1, args.num_samples, total_tokens);
        }
    }

    // Disable hook before cleanup
    engine->ffn_hook = nullptr;
    engine->ffn_hook_data = nullptr;

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
