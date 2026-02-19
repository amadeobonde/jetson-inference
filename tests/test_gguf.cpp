// test_gguf — Unit tests for GGUF parser.
// Creates a minimal synthetic GGUF file and verifies parsing.

#include "jinf/gguf.h"
#include "jinf/quant.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

static bool write_u32(FILE* f, uint32_t v) { return fwrite(&v, 4, 1, f) == 1; }
static bool write_u64(FILE* f, uint64_t v) { return fwrite(&v, 8, 1, f) == 1; }
static bool write_i64(FILE* f, int64_t v)  { return fwrite(&v, 8, 1, f) == 1; }
static bool write_f32(FILE* f, float v)    { return fwrite(&v, 4, 1, f) == 1; }

static bool write_gguf_string(FILE* f, const char* str) {
    uint64_t len = strlen(str);
    return write_u64(f, len) && fwrite(str, 1, len, f) == len;
}

static bool write_kv_u32(FILE* f, const char* key, uint32_t val) {
    return write_gguf_string(f, key) && write_u32(f, 4) && write_u32(f, val);
}

static bool write_kv_f32(FILE* f, const char* key, float val) {
    return write_gguf_string(f, key) && write_u32(f, 6) && write_f32(f, val);
}

static bool write_kv_str(FILE* f, const char* key, const char* val) {
    return write_gguf_string(f, key) && write_u32(f, 8) && write_gguf_string(f, val);
}

// Create a minimal GGUF file with 1 tensor
static const char* create_test_gguf() {
    const char* path = "/tmp/test_jinf.gguf";
    FILE* f = fopen(path, "wb");
    assert(f);

    // Magic + Version
    write_u32(f, 0x46554747);  // "GGUF"
    write_u32(f, 3);           // v3

    // Counts: 1 tensor, 5 KV pairs
    write_u64(f, 1);   // tensor_count
    write_u64(f, 5);   // kv_count

    // KV pairs
    write_kv_str(f, "general.architecture", "llama");
    write_kv_u32(f, "llama.block_count", 2);
    write_kv_u32(f, "llama.embedding_length", 64);
    write_kv_u32(f, "llama.attention.head_count", 4);
    write_kv_f32(f, "llama.attention.layer_norm_rms_epsilon", 1e-5f);

    // Tensor info: token_embd.weight [64, 100] Q4_0
    write_gguf_string(f, "token_embd.weight");
    write_u32(f, 2);       // n_dims
    write_i64(f, 64);      // shape[0]
    write_i64(f, 100);     // shape[1]
    write_u32(f, 2);       // type = Q4_0
    write_u64(f, 0);       // offset within data section

    // Pad to alignment
    long pos = ftell(f);
    long aligned = ((pos + 31) / 32) * 32;
    uint8_t pad[32] = {0};
    fwrite(pad, 1, aligned - pos, f);

    // Write dummy tensor data
    int64_t n_elements = 64 * 100;
    size_t n_bytes = (n_elements / 32) * 18;  // Q4_0: 18 bytes per 32 values
    uint8_t* data = (uint8_t*)calloc(1, n_bytes);
    fwrite(data, 1, n_bytes, f);
    free(data);

    fclose(f);
    return path;
}

int main() {
    printf("=== test_gguf ===\n");

    const char* path = create_test_gguf();

    jinf_gguf_file* file = nullptr;
    jinf_status s = jinf_gguf_open(&file, path);
    assert(s == JINF_OK);
    assert(file != nullptr);

    // Verify header
    assert(file->version == 3);
    assert(file->tensor_count == 1);
    assert(file->kv_count == 5);

    // Verify hyperparameters
    assert(file->hparams.n_layers == 2);
    assert(file->hparams.n_embd == 64);
    assert(file->hparams.n_heads == 4);
    assert(strcmp(file->hparams.arch, "llama") == 0);

    // Verify KV lookup
    const jinf_gguf_kv* kv = jinf_gguf_find_kv(file, "llama.block_count");
    assert(kv != nullptr);
    assert(kv->val_u32 == 2);

    uint32_t val = jinf_gguf_get_u32(file, "llama.embedding_length", 0);
    assert(val == 64);

    // Verify tensor info
    assert(file->tensor_infos[0].n_dims == 2);
    assert(file->tensor_infos[0].shape[0] == 64);
    assert(file->tensor_infos[0].shape[1] == 100);
    assert(file->tensor_infos[0].type == JINF_TYPE_Q4_0);
    assert(file->tensor_infos[0].n_elements == 6400);

    // Verify tensor lookup
    const jinf_gguf_tensor_info* t = jinf_gguf_find_tensor(file, "token_embd.weight");
    assert(t != nullptr);
    assert(t->shape[0] == 64);

    // Verify missing lookups return null
    assert(jinf_gguf_find_kv(file, "nonexistent") == nullptr);
    assert(jinf_gguf_find_tensor(file, "nonexistent") == nullptr);

    jinf_gguf_close(file);

    // Test invalid file
    jinf_gguf_file* bad = nullptr;
    assert(jinf_gguf_open(&bad, "/tmp/nonexistent_file.gguf") != JINF_OK);
    assert(bad == nullptr);

    printf("All GGUF tests passed!\n");
    return 0;
}
