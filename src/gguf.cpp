#include "jinf/gguf.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---- Helpers ----

static bool read_bytes(FILE* f, void* buf, size_t n) {
    return fread(buf, 1, n, f) == n;
}

static bool read_u32(FILE* f, uint32_t* v) { return read_bytes(f, v, 4); }
static bool read_u64(FILE* f, uint64_t* v) { return read_bytes(f, v, 8); }
static bool read_i32(FILE* f, int32_t* v)  { return read_bytes(f, v, 4); }
static bool read_i64(FILE* f, int64_t* v)  { return read_bytes(f, v, 8); }
static bool read_f32(FILE* f, float* v)    { return read_bytes(f, v, 4); }

static bool read_gguf_string(FILE* f, char** out) {
    uint64_t len;
    if (!read_u64(f, &len)) return false;
    if (len > 1024 * 1024) return false;  // sanity limit: 1MB

    char* str = (char*)malloc(len + 1);
    if (!str) return false;

    if (fread(str, 1, len, f) != len) {
        free(str);
        return false;
    }
    str[len] = '\0';
    *out = str;
    return true;
}

static bool skip_gguf_value(FILE* f, jinf_gguf_value_type type);

static bool skip_gguf_array(FILE* f) {
    uint32_t elem_type;
    uint64_t count;
    if (!read_u32(f, &elem_type)) return false;
    if (!read_u64(f, &count)) return false;
    for (uint64_t i = 0; i < count; i++) {
        if (!skip_gguf_value(f, (jinf_gguf_value_type)elem_type)) return false;
    }
    return true;
}

static bool skip_gguf_value(FILE* f, jinf_gguf_value_type type) {
    switch (type) {
        case JINF_GGUF_TYPE_UINT8:
        case JINF_GGUF_TYPE_INT8:
        case JINF_GGUF_TYPE_BOOL:
            return fseek(f, 1, SEEK_CUR) == 0;
        case JINF_GGUF_TYPE_UINT16:
        case JINF_GGUF_TYPE_INT16:
            return fseek(f, 2, SEEK_CUR) == 0;
        case JINF_GGUF_TYPE_UINT32:
        case JINF_GGUF_TYPE_INT32:
        case JINF_GGUF_TYPE_FLOAT32:
            return fseek(f, 4, SEEK_CUR) == 0;
        case JINF_GGUF_TYPE_UINT64:
        case JINF_GGUF_TYPE_INT64:
        case JINF_GGUF_TYPE_FLOAT64:
            return fseek(f, 8, SEEK_CUR) == 0;
        case JINF_GGUF_TYPE_STRING: {
            uint64_t len;
            if (!read_u64(f, &len)) return false;
            return fseek(f, (long)len, SEEK_CUR) == 0;
        }
        case JINF_GGUF_TYPE_ARRAY:
            return skip_gguf_array(f);
        default:
            return false;
    }
}

static bool read_gguf_kv(FILE* f, jinf_gguf_kv* kv) {
    if (!read_gguf_string(f, &kv->key)) return false;

    uint32_t type;
    if (!read_u32(f, &type)) return false;
    kv->type = (jinf_gguf_value_type)type;

    switch (kv->type) {
        case JINF_GGUF_TYPE_UINT8:   return read_bytes(f, &kv->val_u8, 1);
        case JINF_GGUF_TYPE_INT8:    return read_bytes(f, &kv->val_i8, 1);
        case JINF_GGUF_TYPE_UINT16:  return read_bytes(f, &kv->val_u16, 2);
        case JINF_GGUF_TYPE_INT16:   return read_bytes(f, &kv->val_i16, 2);
        case JINF_GGUF_TYPE_UINT32:  return read_bytes(f, &kv->val_u32, 4);
        case JINF_GGUF_TYPE_INT32:   return read_bytes(f, &kv->val_i32, 4);
        case JINF_GGUF_TYPE_FLOAT32: return read_bytes(f, &kv->val_f32, 4);
        case JINF_GGUF_TYPE_BOOL:    return read_bytes(f, &kv->val_bool, 1);
        case JINF_GGUF_TYPE_UINT64:  return read_bytes(f, &kv->val_u64, 8);
        case JINF_GGUF_TYPE_INT64:   return read_bytes(f, &kv->val_i64, 8);
        case JINF_GGUF_TYPE_FLOAT64: return read_bytes(f, &kv->val_f64, 8);
        case JINF_GGUF_TYPE_STRING:
            return read_gguf_string(f, &kv->val_str.data);
        case JINF_GGUF_TYPE_ARRAY: {
            uint32_t et;
            uint64_t cnt;
            if (!read_u32(f, &et)) return false;
            if (!read_u64(f, &cnt)) return false;
            kv->val_arr.elem_type = (jinf_gguf_value_type)et;
            kv->val_arr.count = cnt;
            kv->val_arr.data = nullptr;
            // For array KVs, we skip the data (not needed for hparams)
            for (uint64_t i = 0; i < cnt; i++) {
                if (!skip_gguf_value(f, (jinf_gguf_value_type)et)) return false;
            }
            return true;
        }
        default:
            return false;
    }
}

// Map GGUF tensor type enum to our jinf_qtype
static jinf_qtype gguf_type_to_jinf(uint32_t gguf_type) {
    // GGUF type IDs match ggml types directly
    switch (gguf_type) {
        case 0:  return JINF_TYPE_F32;
        case 1:  return JINF_TYPE_F16;
        case 2:  return JINF_TYPE_Q4_0;
        case 3:  return JINF_TYPE_Q4_1;
        case 6:  return JINF_TYPE_Q5_0;
        case 7:  return JINF_TYPE_Q5_1;
        case 8:  return JINF_TYPE_Q8_0;
        case 9:  return JINF_TYPE_Q8_1;
        case 10: return JINF_TYPE_Q2_K;
        case 11: return JINF_TYPE_Q3_K;
        case 12: return JINF_TYPE_Q4_K;
        case 13: return JINF_TYPE_Q5_K;
        case 14: return JINF_TYPE_Q6_K;
        case 15: return JINF_TYPE_Q8_K;
        default: return JINF_TYPE_F32;
    }
}

// ---- Public API ----

jinf_status jinf_gguf_open(jinf_gguf_file** file_out, const char* path) {
    if (!file_out || !path) return JINF_ERR_INVALID;

    FILE* f = fopen(path, "rb");
    if (!f) {
        JINF_LOG("Failed to open %s: %s", path, strerror(errno));
        return JINF_ERR_IO;
    }

    // Read magic
    uint32_t magic;
    if (!read_u32(f, &magic) || magic != JINF_GGUF_MAGIC) {
        JINF_LOG("Invalid GGUF magic: 0x%08X (expected 0x%08X)", magic, JINF_GGUF_MAGIC);
        fclose(f);
        return JINF_ERR_PARSE;
    }

    // Read version
    uint32_t version;
    if (!read_u32(f, &version)) { fclose(f); return JINF_ERR_PARSE; }
    if (version != JINF_GGUF_VERSION_2 && version != JINF_GGUF_VERSION_3) {
        JINF_LOG("Unsupported GGUF version: %u", version);
        fclose(f);
        return JINF_ERR_PARSE;
    }

    // Read counts
    uint64_t tensor_count, kv_count;
    if (!read_u64(f, &tensor_count)) { fclose(f); return JINF_ERR_PARSE; }
    if (!read_u64(f, &kv_count))     { fclose(f); return JINF_ERR_PARSE; }

    JINF_LOG("GGUF v%u: %llu tensors, %llu KV pairs",
             version, (unsigned long long)tensor_count, (unsigned long long)kv_count);

    // Allocate file struct
    jinf_gguf_file* gf = (jinf_gguf_file*)calloc(1, sizeof(jinf_gguf_file));
    if (!gf) { fclose(f); return JINF_ERR_OOM; }

    gf->version = version;
    gf->tensor_count = tensor_count;
    gf->kv_count = kv_count;
    gf->alignment = JINF_GGUF_DEFAULT_ALIGNMENT;

    // Parse KV pairs
    gf->kvs = (jinf_gguf_kv*)calloc(kv_count, sizeof(jinf_gguf_kv));
    if (!gf->kvs && kv_count > 0) { free(gf); fclose(f); return JINF_ERR_OOM; }

    for (uint64_t i = 0; i < kv_count; i++) {
        if (!read_gguf_kv(f, &gf->kvs[i])) {
            JINF_LOG("Failed to parse KV pair %llu", (unsigned long long)i);
            // Cleanup partial
            jinf_gguf_close(gf);
            fclose(f);
            return JINF_ERR_PARSE;
        }

        // Check for alignment override
        if (strcmp(gf->kvs[i].key, "general.alignment") == 0 &&
            gf->kvs[i].type == JINF_GGUF_TYPE_UINT32) {
            gf->alignment = gf->kvs[i].val_u32;
        }
    }

    // Parse tensor infos
    gf->tensor_infos = (jinf_gguf_tensor_info*)calloc(tensor_count, sizeof(jinf_gguf_tensor_info));
    if (!gf->tensor_infos && tensor_count > 0) {
        jinf_gguf_close(gf);
        fclose(f);
        return JINF_ERR_OOM;
    }

    for (uint64_t i = 0; i < tensor_count; i++) {
        jinf_gguf_tensor_info* ti = &gf->tensor_infos[i];

        if (!read_gguf_string(f, &ti->name)) {
            jinf_gguf_close(gf); fclose(f); return JINF_ERR_PARSE;
        }

        if (!read_u32(f, &ti->n_dims)) {
            jinf_gguf_close(gf); fclose(f); return JINF_ERR_PARSE;
        }

        ti->n_elements = 1;
        for (uint32_t d = 0; d < ti->n_dims; d++) {
            int64_t dim;
            if (!read_i64(f, &dim)) {
                jinf_gguf_close(gf); fclose(f); return JINF_ERR_PARSE;
            }
            ti->shape[d] = dim;
            ti->n_elements *= dim;
        }

        uint32_t type;
        if (!read_u32(f, &type)) {
            jinf_gguf_close(gf); fclose(f); return JINF_ERR_PARSE;
        }
        ti->type = gguf_type_to_jinf(type);

        if (!read_u64(f, &ti->offset)) {
            jinf_gguf_close(gf); fclose(f); return JINF_ERR_PARSE;
        }

        ti->n_bytes = jinf_tensor_nbytes(ti->type, ti->n_elements);
    }

    // Compute data_offset: align current file position to alignment boundary
    long pos = ftell(f);
    uint32_t align = gf->alignment;
    gf->data_offset = (size_t)((pos + align - 1) / align * align);

    fclose(f);

    // Extract hyperparameters from KV pairs
    jinf_model_hparams* hp = &gf->hparams;
    memset(hp, 0, sizeof(*hp));

    // Detect architecture
    const char* arch = jinf_gguf_get_str(gf, "general.architecture", "llama");
    strncpy(hp->arch, arch, sizeof(hp->arch) - 1);

    // Build KV key prefix (e.g., "llama.")
    char prefix[80];
    snprintf(prefix, sizeof(prefix), "%s.", hp->arch);
    size_t plen = strlen(prefix);

    // Helper lambda-like macro for building keys
    char key[256];
    #define GET_HP_U32(field, kv_suffix, def) do { \
        snprintf(key, sizeof(key), "%s%s", prefix, kv_suffix); \
        hp->field = jinf_gguf_get_u32(gf, key, def); \
    } while(0)
    #define GET_HP_F32(field, kv_suffix, def) do { \
        snprintf(key, sizeof(key), "%s%s", prefix, kv_suffix); \
        hp->field = jinf_gguf_get_f32(gf, key, def); \
    } while(0)

    GET_HP_U32(n_layers,     "block_count",           0);
    GET_HP_U32(n_embd,       "embedding_length",      0);
    GET_HP_U32(n_heads,      "attention.head_count",   0);
    GET_HP_U32(n_heads_kv,   "attention.head_count_kv", 0);
    GET_HP_U32(n_ff,         "feed_forward_length",    0);
    GET_HP_U32(n_vocab,      "vocab_size",             0);
    GET_HP_U32(n_ctx_train,  "context_length",         2048);
    GET_HP_F32(rope_freq_base, "rope.freq_base",       10000.0f);
    GET_HP_F32(rms_norm_eps,   "attention.layer_norm_rms_epsilon", 1e-5f);

    #undef GET_HP_U32
    #undef GET_HP_F32

    // If n_heads_kv was not set, default to n_heads (MHA)
    if (hp->n_heads_kv == 0) {
        hp->n_heads_kv = hp->n_heads;
    }

    // If n_vocab was 0, try to get it from the embedding tensor
    if (hp->n_vocab == 0) {
        const jinf_gguf_tensor_info* embd = jinf_gguf_find_tensor(gf, "token_embd.weight");
        if (embd && embd->n_dims >= 2) {
            hp->n_vocab = (uint32_t)embd->shape[1];
        }
    }

    JINF_LOG("Model: %s, layers=%u, embd=%u, heads=%u/%u, ff=%u, vocab=%u",
             hp->arch, hp->n_layers, hp->n_embd, hp->n_heads, hp->n_heads_kv,
             hp->n_ff, hp->n_vocab);

    *file_out = gf;
    return JINF_OK;
}

void jinf_gguf_close(jinf_gguf_file* file) {
    if (!file) return;

    if (file->kvs) {
        for (uint64_t i = 0; i < file->kv_count; i++) {
            free(file->kvs[i].key);
            if (file->kvs[i].type == JINF_GGUF_TYPE_STRING) {
                free(file->kvs[i].val_str.data);
            }
        }
        free(file->kvs);
    }

    if (file->tensor_infos) {
        for (uint64_t i = 0; i < file->tensor_count; i++) {
            free(file->tensor_infos[i].name);
        }
        free(file->tensor_infos);
    }

    free(file);
}

const jinf_gguf_kv* jinf_gguf_find_kv(const jinf_gguf_file* file, const char* key) {
    if (!file || !key) return nullptr;
    for (uint64_t i = 0; i < file->kv_count; i++) {
        if (file->kvs[i].key && strcmp(file->kvs[i].key, key) == 0) {
            return &file->kvs[i];
        }
    }
    return nullptr;
}

const jinf_gguf_tensor_info* jinf_gguf_find_tensor(const jinf_gguf_file* file, const char* name) {
    if (!file || !name) return nullptr;
    for (uint64_t i = 0; i < file->tensor_count; i++) {
        if (file->tensor_infos[i].name && strcmp(file->tensor_infos[i].name, name) == 0) {
            return &file->tensor_infos[i];
        }
    }
    return nullptr;
}

const char* jinf_gguf_get_str(const jinf_gguf_file* file, const char* key, const char* def) {
    const jinf_gguf_kv* kv = jinf_gguf_find_kv(file, key);
    if (!kv || kv->type != JINF_GGUF_TYPE_STRING) return def;
    return kv->val_str.data;
}

uint32_t jinf_gguf_get_u32(const jinf_gguf_file* file, const char* key, uint32_t def) {
    const jinf_gguf_kv* kv = jinf_gguf_find_kv(file, key);
    if (!kv) return def;
    switch (kv->type) {
        case JINF_GGUF_TYPE_UINT32:  return kv->val_u32;
        case JINF_GGUF_TYPE_INT32:   return (uint32_t)kv->val_i32;
        case JINF_GGUF_TYPE_UINT64:  return (uint32_t)kv->val_u64;
        case JINF_GGUF_TYPE_INT64:   return (uint32_t)kv->val_i64;
        case JINF_GGUF_TYPE_FLOAT32: return (uint32_t)kv->val_f32;
        default: return def;
    }
}

float jinf_gguf_get_f32(const jinf_gguf_file* file, const char* key, float def) {
    const jinf_gguf_kv* kv = jinf_gguf_find_kv(file, key);
    if (!kv) return def;
    switch (kv->type) {
        case JINF_GGUF_TYPE_FLOAT32: return kv->val_f32;
        case JINF_GGUF_TYPE_FLOAT64: return (float)kv->val_f64;
        case JINF_GGUF_TYPE_UINT32:  return (float)kv->val_u32;
        case JINF_GGUF_TYPE_INT32:   return (float)kv->val_i32;
        default: return def;
    }
}
