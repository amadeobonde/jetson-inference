// train_predictor — Phase 2: Create a predictor from an activation profile.
//
// MVP mode (static predictor): reads the activation profile from profile_activations
// and creates a predictor file that marks neurons with frequency > threshold as "always active".
//
// Usage: ./train_predictor --profile activations.bin --output predictor.bin [--threshold 0.5]
//
// Future: MLP training mode with --train flag using calibration data.

#include "jinf/activation_predictor.h"
#include "jinf/common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct train_args {
    const char* profile_path;
    const char* output_path;
    float threshold;
};

static bool parse_args(int argc, char** argv, train_args* args) {
    args->profile_path = nullptr;
    args->output_path = "predictor.bin";
    args->threshold = 0.5f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--profile") == 0 && i + 1 < argc) {
            args->profile_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            args->output_path = argv[++i];
        } else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            args->threshold = (float)atof(argv[++i]);
        }
    }

    if (!args->profile_path) {
        fprintf(stderr, "Usage: %s --profile <activations.bin> [--output <predictor.bin>] "
                "[--threshold F]\n", argv[0]);
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    train_args args;
    if (!parse_args(argc, argv, &args)) return 1;

    printf("Creating static predictor from profile: %s\n", args.profile_path);
    printf("Threshold: %.2f\n", args.threshold);

    // Load profile and create static predictor
    jinf_activation_predictor* pred = nullptr;
    jinf_status s = jinf_predictor_create_static(&pred, args.profile_path, args.threshold);
    if (s != JINF_OK) {
        fprintf(stderr, "Failed to create predictor: %s\n", jinf_status_str(s));
        return 1;
    }

    int n_layers = jinf_predictor_n_layers(pred);
    int n_ff = jinf_predictor_n_ff(pred);

    // Print per-layer statistics
    printf("\nPer-layer activation counts:\n");
    int total_active = 0;
    for (int l = 0; l < n_layers; l++) {
        int n_active = 0;
        std::vector<int> ids(n_ff);
        jinf_predictor_predict_static(pred, nullptr, l, ids.data(), &n_active);
        total_active += n_active;
        if (l < 4 || l >= n_layers - 2) {
            printf("  Layer %2d: %5d / %d neurons (%.1f%%)\n",
                   l, n_active, n_ff, 100.0f * n_active / n_ff);
        } else if (l == 4) {
            printf("  ...\n");
        }
    }

    float avg_active = (float)total_active / n_layers;
    printf("\nAverage: %.0f / %d neurons (%.1f%%)\n",
           avg_active, n_ff, 100.0f * avg_active / n_ff);
    printf("Effective speedup: %.1fx less FFN compute\n",
           (float)n_ff / avg_active);

    // For static predictor, the output is the same format as the input profile.
    // The engine loads it via jinf_predictor_create_static at runtime.
    // Just copy the profile as the predictor file.
    FILE* in_fp = fopen(args.profile_path, "rb");
    if (!in_fp) {
        fprintf(stderr, "Failed to reopen profile\n");
        jinf_predictor_destroy(pred);
        return 1;
    }

    FILE* out_fp = fopen(args.output_path, "wb");
    if (!out_fp) {
        fprintf(stderr, "Failed to create output: %s\n", args.output_path);
        fclose(in_fp);
        jinf_predictor_destroy(pred);
        return 1;
    }

    // Copy profile to output
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), in_fp)) > 0) {
        fwrite(buf, 1, n, out_fp);
    }

    fclose(in_fp);
    fclose(out_fp);

    printf("\nPredictor saved to: %s\n", args.output_path);

    jinf_predictor_destroy(pred);
    return 0;
}
