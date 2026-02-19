#pragma once

#include <cstdint>
#include <cstddef>

// ---- Quantization types ----

enum jinf_qtype : int {
    JINF_TYPE_F32   = 0,
    JINF_TYPE_F16   = 1,
    JINF_TYPE_Q4_0  = 2,
    JINF_TYPE_Q4_1  = 3,
    JINF_TYPE_Q5_0  = 6,
    JINF_TYPE_Q5_1  = 7,
    JINF_TYPE_Q8_0  = 8,
    JINF_TYPE_Q8_1  = 9,
    JINF_TYPE_Q2_K  = 10,
    JINF_TYPE_Q3_K  = 11,
    JINF_TYPE_Q4_K  = 12,
    JINF_TYPE_Q5_K  = 13,
    JINF_TYPE_Q6_K  = 14,
    JINF_TYPE_Q8_K  = 15,
    JINF_TYPE_COUNT,
};

// ---- Binary block layouts (matching ggml exactly) ----

// Q4_0: 18 bytes for 32 values
// Layout: half-precision scale (d) + 16 bytes of packed 4-bit quantized values
#pragma pack(push, 1)
struct jinf_block_q4_0 {
    uint16_t d;         // delta (fp16)
    uint8_t  qs[16];   // quantized values (4 bits each, 32 values)
};
static_assert(sizeof(jinf_block_q4_0) == 18, "Q4_0 block size mismatch");
#pragma pack(pop)

// Q4_K_M: 144 bytes for 256 values (K-quant super-block)
// Layout: 2x fp16 scales + 12 bytes sub-block scales + 128 bytes packed nibbles
#pragma pack(push, 1)
struct jinf_block_q4_K {
    uint16_t d;         // super-block scale (fp16)
    uint16_t dmin;      // super-block minimum (fp16)
    uint8_t  scales[12]; // sub-block scales and mins (6-bit packed)
    uint8_t  qs[128];   // quantized values (4 bits each, 256 values)
};
static_assert(sizeof(jinf_block_q4_K) == 144, "Q4_K block size mismatch");
#pragma pack(pop)

// Q8_0: 34 bytes for 32 values
#pragma pack(push, 1)
struct jinf_block_q8_0 {
    uint16_t d;         // delta (fp16)
    int8_t   qs[32];   // quantized values
};
static_assert(sizeof(jinf_block_q8_0) == 34, "Q8_0 block size mismatch");
#pragma pack(pop)

// ---- Block size / type size queries ----

// Number of values per quantization block
inline int jinf_qtype_block_size(jinf_qtype type) {
    switch (type) {
        case JINF_TYPE_F32:   return 1;
        case JINF_TYPE_F16:   return 1;
        case JINF_TYPE_Q4_0:  return 32;
        case JINF_TYPE_Q4_1:  return 32;
        case JINF_TYPE_Q5_0:  return 32;
        case JINF_TYPE_Q5_1:  return 32;
        case JINF_TYPE_Q8_0:  return 32;
        case JINF_TYPE_Q8_1:  return 32;
        case JINF_TYPE_Q2_K:  return 256;
        case JINF_TYPE_Q3_K:  return 256;
        case JINF_TYPE_Q4_K:  return 256;
        case JINF_TYPE_Q5_K:  return 256;
        case JINF_TYPE_Q6_K:  return 256;
        case JINF_TYPE_Q8_K:  return 256;
        default:              return 0;
    }
}

// Byte size per quantization block
inline size_t jinf_qtype_type_size(jinf_qtype type) {
    switch (type) {
        case JINF_TYPE_F32:   return 4;
        case JINF_TYPE_F16:   return 2;
        case JINF_TYPE_Q4_0:  return sizeof(jinf_block_q4_0);  // 18
        case JINF_TYPE_Q4_1:  return 20;
        case JINF_TYPE_Q5_0:  return 22;
        case JINF_TYPE_Q5_1:  return 24;
        case JINF_TYPE_Q8_0:  return sizeof(jinf_block_q8_0);  // 34
        case JINF_TYPE_Q8_1:  return 36;
        case JINF_TYPE_Q2_K:  return 84;
        case JINF_TYPE_Q3_K:  return 110;
        case JINF_TYPE_Q4_K:  return sizeof(jinf_block_q4_K);  // 144
        case JINF_TYPE_Q5_K:  return 176;
        case JINF_TYPE_Q6_K:  return 210;
        case JINF_TYPE_Q8_K:  return 292;
        default:              return 0;
    }
}

inline const char* jinf_qtype_name(jinf_qtype type) {
    switch (type) {
        case JINF_TYPE_F32:   return "F32";
        case JINF_TYPE_F16:   return "F16";
        case JINF_TYPE_Q4_0:  return "Q4_0";
        case JINF_TYPE_Q4_1:  return "Q4_1";
        case JINF_TYPE_Q5_0:  return "Q5_0";
        case JINF_TYPE_Q5_1:  return "Q5_1";
        case JINF_TYPE_Q8_0:  return "Q8_0";
        case JINF_TYPE_Q8_1:  return "Q8_1";
        case JINF_TYPE_Q2_K:  return "Q2_K";
        case JINF_TYPE_Q3_K:  return "Q3_K";
        case JINF_TYPE_Q4_K:  return "Q4_K";
        case JINF_TYPE_Q5_K:  return "Q5_K";
        case JINF_TYPE_Q6_K:  return "Q6_K";
        case JINF_TYPE_Q8_K:  return "Q8_K";
        default:              return "UNKNOWN";
    }
}

// Total bytes for a tensor with n_elements values of given type
inline size_t jinf_tensor_nbytes(jinf_qtype type, int64_t n_elements) {
    int block_size = jinf_qtype_block_size(type);
    if (block_size == 0) return 0;
    size_t n_blocks = (size_t)((n_elements + block_size - 1) / block_size);
    return n_blocks * jinf_qtype_type_size(type);
}

// ---- CPU dequantization (reference implementations) ----

void jinf_dequantize_q4_0(const jinf_block_q4_0* block, float* out, int n_blocks);
void jinf_dequantize_q4_K(const jinf_block_q4_K* block, float* out, int n_blocks);
void jinf_dequantize_q8_0(const jinf_block_q8_0* block, float* out, int n_blocks);

// Generic dequantize dispatcher
void jinf_dequantize(const void* data, jinf_qtype type, float* out, int64_t n_elements);
