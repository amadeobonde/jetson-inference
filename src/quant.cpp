#include "jinf/quant.h"
#include <cmath>
#include <cstring>

// ---- fp16 conversion helpers ----

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x03FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            uint32_t result = sign;
            float f;
            memcpy(&f, &result, 4);
            return f;
        }
        // Denormalized
        exponent = 1;
        while (!(mantissa & 0x0400)) {
            mantissa <<= 1;
            exponent--;
        }
        mantissa &= 0x03FF;
        exponent = exponent + (127 - 15);
        uint32_t result = sign | (exponent << 23) | (mantissa << 13);
        float f;
        memcpy(&f, &result, 4);
        return f;
    } else if (exponent == 31) {
        // Inf or NaN
        uint32_t result = sign | 0x7F800000 | (mantissa << 13);
        float f;
        memcpy(&f, &result, 4);
        return f;
    }

    exponent = exponent + (127 - 15);
    uint32_t result = sign | (exponent << 23) | (mantissa << 13);
    float f;
    memcpy(&f, &result, 4);
    return f;
}

// ---- Q4_0 dequantization ----

void jinf_dequantize_q4_0(const jinf_block_q4_0* block, float* out, int n_blocks) {
    for (int b = 0; b < n_blocks; b++) {
        float d = fp16_to_fp32(block[b].d);
        for (int i = 0; i < 16; i++) {
            uint8_t byte = block[b].qs[i];
            int8_t v0 = (int8_t)(byte & 0x0F) - 8;
            int8_t v1 = (int8_t)(byte >> 4) - 8;
            out[b * 32 + i * 2 + 0] = (float)v0 * d;
            out[b * 32 + i * 2 + 1] = (float)v1 * d;
        }
    }
}

// ---- Q4_K dequantization ----

void jinf_dequantize_q4_K(const jinf_block_q4_K* block, float* out, int n_blocks) {
    for (int b = 0; b < n_blocks; b++) {
        float d = fp16_to_fp32(block[b].d);
        float dmin = fp16_to_fp32(block[b].dmin);

        // Decode 6-bit sub-block scales and minimums from the 12-byte scales array.
        // There are 8 sub-blocks of 32 values each (256 total).
        // The 12-byte scales array packs 8 scales + 8 mins at 6 bits each = 96 bits = 12 bytes.
        uint8_t sc[8], mn[8];
        for (int i = 0; i < 4; i++) {
            sc[i]     = block[b].scales[i] & 0x3F;
            mn[i]     = block[b].scales[i + 4] & 0x3F;
            sc[i + 4] = ((block[b].scales[i + 8] & 0x0F) << 2) | (block[b].scales[i] >> 6);
            mn[i + 4] = ((block[b].scales[i + 8] >> 4)   << 2) | (block[b].scales[i + 4] >> 6);
        }

        for (int sb = 0; sb < 8; sb++) {
            float sub_d   = d * (float)sc[sb];
            float sub_min = dmin * (float)mn[sb];
            int offset = sb * 32;
            // Each sub-block has 32 values stored as 16 bytes of packed nibbles
            int qs_off = sb * 16;
            for (int i = 0; i < 16; i++) {
                uint8_t byte = block[b].qs[qs_off + i];
                out[b * 256 + offset + i * 2 + 0] = sub_d * (float)(byte & 0x0F) - sub_min;
                out[b * 256 + offset + i * 2 + 1] = sub_d * (float)(byte >> 4)    - sub_min;
            }
        }
    }
}

// ---- Q8_0 dequantization ----

void jinf_dequantize_q8_0(const jinf_block_q8_0* block, float* out, int n_blocks) {
    for (int b = 0; b < n_blocks; b++) {
        float d = fp16_to_fp32(block[b].d);
        for (int i = 0; i < 32; i++) {
            out[b * 32 + i] = (float)block[b].qs[i] * d;
        }
    }
}

// ---- Generic dispatcher ----

void jinf_dequantize(const void* data, jinf_qtype type, float* out, int64_t n_elements) {
    int block_size = jinf_qtype_block_size(type);
    if (block_size == 0) return;
    int n_blocks = (int)((n_elements + block_size - 1) / block_size);

    switch (type) {
        case JINF_TYPE_Q4_0:
            jinf_dequantize_q4_0((const jinf_block_q4_0*)data, out, n_blocks);
            break;
        case JINF_TYPE_Q4_K:
            jinf_dequantize_q4_K((const jinf_block_q4_K*)data, out, n_blocks);
            break;
        case JINF_TYPE_Q8_0:
            jinf_dequantize_q8_0((const jinf_block_q8_0*)data, out, n_blocks);
            break;
        case JINF_TYPE_F32:
            memcpy(out, data, n_elements * sizeof(float));
            break;
        case JINF_TYPE_F16: {
            const uint16_t* fp16 = (const uint16_t*)data;
            for (int64_t i = 0; i < n_elements; i++) {
                out[i] = fp16_to_fp32(fp16[i]);
            }
            break;
        }
        default:
            break;
    }
}
