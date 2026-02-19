#pragma once

#include "jinf/common.h"
#include "jinf/quant.h"
#include <cstdint>
#include <cstddef>

// ---- .nvmw file format ----
//
// All offsets are 4K-aligned for O_DIRECT compatibility.
//
// [Header: 4KB]
// [Layer Index: 4K-aligned]
// [Hot Region: 4K-aligned]         all hot tensors packed contiguously
// [Cold Layer 0: 4K-aligned]       cold tensors packed contiguously per layer
// [Cold Layer 1: 4K-aligned]
// ...

#define JINF_NVMW_MAGIC           0x574D564E  // "NVMW" in little-endian
#define JINF_NVMW_VERSION         1
#define JINF_NVMW_VERSION_BUNDLED 2

// ---- Per-tensor entry in the layer index ----

struct jinf_nvmw_tensor_entry {
    char     name[128];
    jinf_qtype type;
    int64_t  shape[4];
    uint32_t n_dims;
    uint64_t offset;       // offset within the region (hot or cold layer)
    uint64_t n_bytes;
    int32_t  layer_index;  // -1 for non-layer tensors (embeddings, output)
    uint8_t  is_hot;       // 1 = hot region, 0 = cold region
    uint8_t  _pad[3];
};

// ---- Per-layer cold data descriptor ----

struct jinf_nvmw_layer_desc {
    uint64_t cold_start_offset;  // absolute file offset to this layer's cold data
    uint64_t cold_total_size;    // total bytes of cold data for this layer
    uint32_t n_cold_tensors;
    uint32_t _pad;
};

// ---- Neuron bundle structures (Phase 2: Sparse FFN) ----
//
// Bundle layout per neuron: gate_row + up_row + down_col in original quant format,
// padded to 4K alignment for O_DIRECT compatibility.

#pragma pack(push, 1)
struct jinf_nvmw_bundle_header {
    uint32_t n_bundle_layers;      // number of layers with bundles
    uint32_t bundle_size;          // bytes per neuron bundle (padded to 4K)
    uint32_t n_ff;                 // neurons per layer
    uint32_t _pad;
};

struct jinf_nvmw_bundle_layer_desc {
    uint64_t bundles_offset;       // absolute file offset to this layer's bundles
    uint64_t bundles_total_size;   // n_ff * bundle_size
    uint32_t n_neurons;            // == n_ff
    uint32_t bundle_size;          // bytes per bundle for this layer
};
#pragma pack(pop)

// ---- File header (fits in 4KB) ----

#pragma pack(push, 1)
struct jinf_nvmw_header {
    uint32_t magic;              // 4
    uint32_t version;            // 4
    uint32_t n_layers;           // 4
    uint32_t n_tensors;          // 4

    // Offsets and sizes for the main regions
    uint64_t index_offset;       // 8
    uint64_t index_size;         // 8
    uint64_t hot_offset;         // 8
    uint64_t hot_size;           // 8
    uint64_t cold_start_offset;  // 8
    uint64_t cold_total_size;    // 8

    // Model hyperparameters (copied from GGUF)
    uint32_t n_embd;             // 4
    uint32_t n_heads;            // 4
    uint32_t n_heads_kv;         // 4
    uint32_t n_ff;               // 4
    uint32_t n_vocab;            // 4
    uint32_t n_ctx_train;        // 4
    float    rope_freq_base;     // 4
    float    rms_norm_eps;       // 4
    int32_t  primary_type;       // 4 (jinf_qtype value)
    uint32_t _pad0;              // 4

    // Phase 2: bundle index offset (0 = no bundles, backwards compatible)
    uint64_t bundle_index_offset;   // 8

    // Total: 104 + 8 = 112
    uint8_t  _reserved[4096 - 112];
};
#pragma pack(pop)
static_assert(sizeof(jinf_nvmw_header) == 4096, "NVMW header must be 4KB");

// ---- Opaque reader ----

struct jinf_nvmw_reader;

// ---- Writer API ----

struct jinf_nvmw_writer;

jinf_status jinf_nvmw_writer_create(jinf_nvmw_writer** w, const char* path);
void        jinf_nvmw_writer_destroy(jinf_nvmw_writer* w);
jinf_status jinf_nvmw_writer_set_header(jinf_nvmw_writer* w, const jinf_nvmw_header* hdr);
jinf_status jinf_nvmw_writer_add_layer_desc(jinf_nvmw_writer* w, int layer, const jinf_nvmw_layer_desc* desc);
jinf_status jinf_nvmw_writer_add_tensor_entry(jinf_nvmw_writer* w, const jinf_nvmw_tensor_entry* entry);
jinf_status jinf_nvmw_writer_write_hot(jinf_nvmw_writer* w, const void* data, size_t size);
jinf_status jinf_nvmw_writer_write_cold_layer(jinf_nvmw_writer* w, int layer, const void* data, size_t size);
jinf_status jinf_nvmw_writer_finalize(jinf_nvmw_writer* w);

// ---- Reader API ----

jinf_status jinf_nvmw_open(jinf_nvmw_reader** r, const char* path);
void        jinf_nvmw_close(jinf_nvmw_reader* r);

const jinf_nvmw_header*       jinf_nvmw_get_header(const jinf_nvmw_reader* r);
const jinf_nvmw_layer_desc*   jinf_nvmw_get_layer_desc(const jinf_nvmw_reader* r, int layer);
const jinf_nvmw_tensor_entry* jinf_nvmw_get_tensor_entries(const jinf_nvmw_reader* r, int* count);

jinf_status jinf_nvmw_get_layer_cold_range(const jinf_nvmw_reader* r, int layer,
                                            uint64_t* offset, uint64_t* size);
jinf_status jinf_nvmw_get_hot_range(const jinf_nvmw_reader* r, uint64_t* offset, uint64_t* size);

// Find a tensor entry by name. Returns nullptr if not found.
const jinf_nvmw_tensor_entry* jinf_nvmw_find_tensor(const jinf_nvmw_reader* r, const char* name);

// ---- Bundle reader API (Phase 2) ----

// Check if bundles are present in this file (version >= 2, bundle_index_offset != 0)
bool jinf_nvmw_has_bundles(const jinf_nvmw_reader* r);

// Get bundle header. Returns nullptr if no bundles.
const jinf_nvmw_bundle_header* jinf_nvmw_get_bundle_header(const jinf_nvmw_reader* r);

// Get bundle layer descriptor for a given layer. Returns nullptr if out of range.
const jinf_nvmw_bundle_layer_desc* jinf_nvmw_get_bundle_layer_desc(const jinf_nvmw_reader* r, int layer);

// Get the absolute file offset for a specific neuron bundle in a given layer.
uint64_t jinf_nvmw_get_neuron_bundle_offset(const jinf_nvmw_reader* r, int layer, int neuron_id);

// ---- Writer bundle API (Phase 2) ----

jinf_status jinf_nvmw_writer_write_bundle_section(jinf_nvmw_writer* w,
    const jinf_nvmw_bundle_header* bh,
    const jinf_nvmw_bundle_layer_desc* layer_descs,
    int n_layers);
