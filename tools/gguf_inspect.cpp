// gguf_inspect — Dump GGUF file metadata and tensor listing.
// Usage: ./gguf_inspect <model.gguf>

#include "jinf/gguf.h"
#include "jinf/quant.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void print_kv(const jinf_gguf_kv* kv) {
    printf("  %-50s  ", kv->key);
    switch (kv->type) {
        case JINF_GGUF_TYPE_UINT8:   printf("u8     = %u\n", kv->val_u8); break;
        case JINF_GGUF_TYPE_INT8:    printf("i8     = %d\n", kv->val_i8); break;
        case JINF_GGUF_TYPE_UINT16:  printf("u16    = %u\n", kv->val_u16); break;
        case JINF_GGUF_TYPE_INT16:   printf("i16    = %d\n", kv->val_i16); break;
        case JINF_GGUF_TYPE_UINT32:  printf("u32    = %u\n", kv->val_u32); break;
        case JINF_GGUF_TYPE_INT32:   printf("i32    = %d\n", kv->val_i32); break;
        case JINF_GGUF_TYPE_FLOAT32: printf("f32    = %.6f\n", kv->val_f32); break;
        case JINF_GGUF_TYPE_BOOL:    printf("bool   = %s\n", kv->val_bool ? "true" : "false"); break;
        case JINF_GGUF_TYPE_UINT64:  printf("u64    = %llu\n", (unsigned long long)kv->val_u64); break;
        case JINF_GGUF_TYPE_INT64:   printf("i64    = %lld\n", (long long)kv->val_i64); break;
        case JINF_GGUF_TYPE_FLOAT64: printf("f64    = %.6f\n", kv->val_f64); break;
        case JINF_GGUF_TYPE_STRING:
            printf("str    = \"%.*s\"\n",
                   (int)(kv->val_str.length > 80 ? 80 : kv->val_str.length),
                   kv->val_str.data);
            break;
        case JINF_GGUF_TYPE_ARRAY:
            printf("arr[%llu] of type %d\n",
                   (unsigned long long)kv->val_arr.count, kv->val_arr.elem_type);
            break;
        default:
            printf("unknown type %d\n", kv->type);
            break;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    jinf_gguf_file* file = nullptr;
    jinf_status s = jinf_gguf_open(&file, argv[1]);
    if (s != JINF_OK) {
        fprintf(stderr, "Failed to open %s: %s\n", argv[1], jinf_status_str(s));
        return 1;
    }

    printf("=== GGUF File: %s ===\n", argv[1]);
    printf("Version:       %u\n", file->version);
    printf("Tensor count:  %llu\n", (unsigned long long)file->tensor_count);
    printf("KV count:      %llu\n", (unsigned long long)file->kv_count);
    printf("Alignment:     %u\n", file->alignment);
    printf("Data offset:   0x%llX (%llu bytes)\n",
           (unsigned long long)file->data_offset, (unsigned long long)file->data_offset);

    // Hyperparameters
    printf("\n=== Hyperparameters ===\n");
    printf("  Architecture:    %s\n", file->hparams.arch);
    printf("  Layers:          %u\n", file->hparams.n_layers);
    printf("  Embedding dim:   %u\n", file->hparams.n_embd);
    printf("  Heads:           %u\n", file->hparams.n_heads);
    printf("  KV Heads:        %u\n", file->hparams.n_heads_kv);
    printf("  FF size:         %u\n", file->hparams.n_ff);
    printf("  Vocab size:      %u\n", file->hparams.n_vocab);
    printf("  Context (train): %u\n", file->hparams.n_ctx_train);
    printf("  RoPE freq base:  %.1f\n", file->hparams.rope_freq_base);
    printf("  RMS norm eps:    %.1e\n", file->hparams.rms_norm_eps);

    // KV pairs
    printf("\n=== KV Pairs (%llu) ===\n", (unsigned long long)file->kv_count);
    for (uint64_t i = 0; i < file->kv_count; i++) {
        print_kv(&file->kvs[i]);
    }

    // Tensor listing
    printf("\n=== Tensors (%llu) ===\n", (unsigned long long)file->tensor_count);
    size_t total_bytes = 0;
    for (uint64_t i = 0; i < file->tensor_count; i++) {
        const jinf_gguf_tensor_info* t = &file->tensor_infos[i];
        printf("  %-50s  %6s  [", t->name, jinf_qtype_name(t->type));
        for (uint32_t d = 0; d < t->n_dims; d++) {
            if (d > 0) printf(", ");
            printf("%lld", (long long)t->shape[d]);
        }
        printf("]  %8.2f MB  offset=0x%llX\n",
               t->n_bytes / (1024.0 * 1024.0),
               (unsigned long long)t->offset);
        total_bytes += t->n_bytes;
    }

    printf("\n=== Summary ===\n");
    printf("  Total tensor data: %.2f MB (%.2f GB)\n",
           total_bytes / (1024.0 * 1024.0),
           total_bytes / (1024.0 * 1024.0 * 1024.0));

    jinf_gguf_close(file);
    return 0;
}
