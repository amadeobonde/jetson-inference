#pragma once

#include "jinf/common.h"
#include "jinf/quant.h"
#include <cstdint>
#include <cstddef>

// ---- GGUF constants ----

#define JINF_GGUF_MAGIC      0x46554747  // "GGUF" in little-endian
#define JINF_GGUF_VERSION_2  2
#define JINF_GGUF_VERSION_3  3
#define JINF_GGUF_DEFAULT_ALIGNMENT 32

// ---- GGUF value types ----

enum jinf_gguf_value_type : int {
    JINF_GGUF_TYPE_UINT8   = 0,
    JINF_GGUF_TYPE_INT8    = 1,
    JINF_GGUF_TYPE_UINT16  = 2,
    JINF_GGUF_TYPE_INT16   = 3,
    JINF_GGUF_TYPE_UINT32  = 4,
    JINF_GGUF_TYPE_INT32   = 5,
    JINF_GGUF_TYPE_FLOAT32 = 6,
    JINF_GGUF_TYPE_BOOL    = 7,
    JINF_GGUF_TYPE_STRING  = 8,
    JINF_GGUF_TYPE_ARRAY   = 9,
    JINF_GGUF_TYPE_UINT64  = 10,
    JINF_GGUF_TYPE_INT64   = 11,
    JINF_GGUF_TYPE_FLOAT64 = 12,
};

// ---- GGUF KV pair ----

struct jinf_gguf_kv {
    char*                  key;
    jinf_gguf_value_type   type;
    union {
        uint8_t   val_u8;
        int8_t    val_i8;
        uint16_t  val_u16;
        int16_t   val_i16;
        uint32_t  val_u32;
        int32_t   val_i32;
        float     val_f32;
        bool      val_bool;
        uint64_t  val_u64;
        int64_t   val_i64;
        double    val_f64;
        struct { char* data; uint64_t length; } val_str;
        struct { jinf_gguf_value_type elem_type; uint64_t count; void* data; } val_arr;
    };
};

// ---- Tensor info ----

struct jinf_gguf_tensor_info {
    char*      name;
    uint32_t   n_dims;
    int64_t    shape[4];
    jinf_qtype type;
    uint64_t   offset;      // offset within the data section
    int64_t    n_elements;
    size_t     n_bytes;
};

// ---- Model hyperparameters (extracted from KV) ----

struct jinf_model_hparams {
    char     arch[64];
    uint32_t n_layers;
    uint32_t n_embd;
    uint32_t n_heads;
    uint32_t n_heads_kv;
    uint32_t n_ff;
    uint32_t n_vocab;
    uint32_t n_ctx_train;
    float    rope_freq_base;
    float    rms_norm_eps;
};

// ---- Parsed GGUF file ----

struct jinf_gguf_file {
    uint32_t             version;
    uint64_t             tensor_count;
    uint64_t             kv_count;
    uint32_t             alignment;
    jinf_gguf_kv*        kvs;
    jinf_gguf_tensor_info* tensor_infos;
    size_t               data_offset;   // absolute file offset to tensor data
    jinf_model_hparams   hparams;
};

// ---- API ----

jinf_status jinf_gguf_open(jinf_gguf_file** file, const char* path);
void        jinf_gguf_close(jinf_gguf_file* file);

// Find a KV pair by key. Returns nullptr if not found.
const jinf_gguf_kv* jinf_gguf_find_kv(const jinf_gguf_file* file, const char* key);

// Find a tensor info by name. Returns nullptr if not found.
const jinf_gguf_tensor_info* jinf_gguf_find_tensor(const jinf_gguf_file* file, const char* name);

// Get string value from KV, or default if not found.
const char* jinf_gguf_get_str(const jinf_gguf_file* file, const char* key, const char* def);

// Get uint32 value from KV, or default.
uint32_t jinf_gguf_get_u32(const jinf_gguf_file* file, const char* key, uint32_t def);

// Get float value from KV, or default.
float jinf_gguf_get_f32(const jinf_gguf_file* file, const char* key, float def);
