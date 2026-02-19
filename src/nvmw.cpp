#include "jinf/nvmw.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---- Writer implementation ----

struct jinf_nvmw_writer {
    FILE* fp;
    jinf_nvmw_header header;
    std::vector<jinf_nvmw_layer_desc> layer_descs;
    std::vector<jinf_nvmw_tensor_entry> tensor_entries;
    size_t hot_written;
    size_t current_cold_offset;
};

jinf_status jinf_nvmw_writer_create(jinf_nvmw_writer** w, const char* path) {
    if (!w || !path) return JINF_ERR_INVALID;

    FILE* fp = fopen(path, "wb");
    if (!fp) {
        JINF_LOG("Failed to create %s", path);
        return JINF_ERR_IO;
    }

    jinf_nvmw_writer* wr = new (std::nothrow) jinf_nvmw_writer();
    if (!wr) { fclose(fp); return JINF_ERR_OOM; }

    wr->fp = fp;
    memset(&wr->header, 0, sizeof(wr->header));
    wr->hot_written = 0;
    wr->current_cold_offset = 0;

    *w = wr;
    return JINF_OK;
}

void jinf_nvmw_writer_destroy(jinf_nvmw_writer* w) {
    if (!w) return;
    if (w->fp) fclose(w->fp);
    delete w;
}

jinf_status jinf_nvmw_writer_set_header(jinf_nvmw_writer* w, const jinf_nvmw_header* hdr) {
    if (!w || !hdr) return JINF_ERR_INVALID;
    memcpy(&w->header, hdr, sizeof(jinf_nvmw_header));
    return JINF_OK;
}

jinf_status jinf_nvmw_writer_add_layer_desc(jinf_nvmw_writer* w, int layer,
                                              const jinf_nvmw_layer_desc* desc) {
    if (!w || !desc) return JINF_ERR_INVALID;
    if (layer >= (int)w->layer_descs.size()) {
        w->layer_descs.resize(layer + 1);
    }
    w->layer_descs[layer] = *desc;
    return JINF_OK;
}

jinf_status jinf_nvmw_writer_add_tensor_entry(jinf_nvmw_writer* w,
                                                const jinf_nvmw_tensor_entry* entry) {
    if (!w || !entry) return JINF_ERR_INVALID;
    w->tensor_entries.push_back(*entry);
    return JINF_OK;
}

// Pad file to 4K boundary
static jinf_status pad_to_4k(FILE* fp) {
    long pos = ftell(fp);
    if (pos < 0) return JINF_ERR_IO;
    size_t aligned = JINF_ALIGN_4K((size_t)pos);
    size_t pad = aligned - (size_t)pos;
    if (pad > 0) {
        uint8_t zeros[4096] = {0};
        while (pad > 0) {
            size_t chunk = pad > 4096 ? 4096 : pad;
            if (fwrite(zeros, 1, chunk, fp) != chunk) return JINF_ERR_IO;
            pad -= chunk;
        }
    }
    return JINF_OK;
}

jinf_status jinf_nvmw_writer_write_hot(jinf_nvmw_writer* w, const void* data, size_t size) {
    if (!w || !data) return JINF_ERR_INVALID;
    if (fwrite(data, 1, size, w->fp) != size) return JINF_ERR_IO;
    w->hot_written += size;
    return JINF_OK;
}

jinf_status jinf_nvmw_writer_write_cold_layer(jinf_nvmw_writer* w, int layer,
                                                const void* data, size_t size) {
    if (!w || !data) return JINF_ERR_INVALID;

    // Pad to 4K before writing cold layer
    jinf_status s = pad_to_4k(w->fp);
    if (s != JINF_OK) return s;

    long pos = ftell(w->fp);
    if (layer < (int)w->layer_descs.size()) {
        w->layer_descs[layer].cold_start_offset = (uint64_t)pos;
        w->layer_descs[layer].cold_total_size = size;
    }

    if (fwrite(data, 1, size, w->fp) != size) return JINF_ERR_IO;
    return JINF_OK;
}

jinf_status jinf_nvmw_writer_finalize(jinf_nvmw_writer* w) {
    if (!w || !w->fp) return JINF_ERR_INVALID;

    // Compute layout:
    // [Header: 4KB]
    // [Layer Index + Tensor Entries: 4K-aligned]
    // [Hot Region: already written after this was called in order]
    // [Cold Layers: already written]
    //
    // Actually, the caller writes data in order, so we need to assemble the file
    // with a seek-back approach. Instead, let's plan the layout:

    // For simplicity, rewrite the file with correct offsets:
    // The caller is expected to:
    //   1. Create writer
    //   2. Set header
    //   3. Add tensor entries
    //   4. Finalize will write: header, index, then caller writes hot+cold

    // We'll write header at offset 0
    rewind(w->fp);

    // Compute index region
    size_t index_size = w->layer_descs.size() * sizeof(jinf_nvmw_layer_desc)
                      + w->tensor_entries.size() * sizeof(jinf_nvmw_tensor_entry);

    w->header.magic = JINF_NVMW_MAGIC;
    w->header.version = JINF_NVMW_VERSION;
    w->header.n_tensors = (uint32_t)w->tensor_entries.size();
    w->header.index_offset = 4096;  // right after header
    w->header.index_size = index_size;

    size_t hot_offset = JINF_ALIGN_4K(4096 + index_size);
    w->header.hot_offset = hot_offset;
    // hot_size, cold_start_offset, cold_total_size set by caller or computed

    // Write header
    if (fwrite(&w->header, sizeof(jinf_nvmw_header), 1, w->fp) != 1) return JINF_ERR_IO;

    // Write index: layer descriptors, then tensor entries
    for (auto& ld : w->layer_descs) {
        if (fwrite(&ld, sizeof(jinf_nvmw_layer_desc), 1, w->fp) != 1) return JINF_ERR_IO;
    }
    for (auto& te : w->tensor_entries) {
        if (fwrite(&te, sizeof(jinf_nvmw_tensor_entry), 1, w->fp) != 1) return JINF_ERR_IO;
    }

    // Pad to hot_offset
    jinf_status s = pad_to_4k(w->fp);
    if (s != JINF_OK) return s;

    fflush(w->fp);
    return JINF_OK;
}

// ---- Reader implementation ----

struct jinf_nvmw_reader {
    jinf_nvmw_header header;
    std::vector<jinf_nvmw_layer_desc> layer_descs;
    std::vector<jinf_nvmw_tensor_entry> tensor_entries;
    char path[1024];

    // Phase 2: bundle data (populated if bundle_index_offset != 0)
    bool has_bundles;
    jinf_nvmw_bundle_header bundle_header;
    std::vector<jinf_nvmw_bundle_layer_desc> bundle_layer_descs;
};

jinf_status jinf_nvmw_open(jinf_nvmw_reader** r, const char* path) {
    if (!r || !path) return JINF_ERR_INVALID;

    FILE* fp = fopen(path, "rb");
    if (!fp) {
        JINF_LOG("Failed to open %s", path);
        return JINF_ERR_IO;
    }

    jinf_nvmw_reader* rd = new (std::nothrow) jinf_nvmw_reader();
    if (!rd) { fclose(fp); return JINF_ERR_OOM; }

    strncpy(rd->path, path, sizeof(rd->path) - 1);

    // Read header
    if (fread(&rd->header, sizeof(jinf_nvmw_header), 1, fp) != 1) {
        delete rd; fclose(fp); return JINF_ERR_PARSE;
    }

    if (rd->header.magic != JINF_NVMW_MAGIC) {
        JINF_LOG("Invalid NVMW magic: 0x%08X", rd->header.magic);
        delete rd; fclose(fp); return JINF_ERR_PARSE;
    }

    // Read index: seek to index_offset
    fseek(fp, (long)rd->header.index_offset, SEEK_SET);

    // Read layer descriptors
    rd->layer_descs.resize(rd->header.n_layers);
    for (uint32_t i = 0; i < rd->header.n_layers; i++) {
        if (fread(&rd->layer_descs[i], sizeof(jinf_nvmw_layer_desc), 1, fp) != 1) {
            delete rd; fclose(fp); return JINF_ERR_PARSE;
        }
    }

    // Read tensor entries
    rd->tensor_entries.resize(rd->header.n_tensors);
    for (uint32_t i = 0; i < rd->header.n_tensors; i++) {
        if (fread(&rd->tensor_entries[i], sizeof(jinf_nvmw_tensor_entry), 1, fp) != 1) {
            delete rd; fclose(fp); return JINF_ERR_PARSE;
        }
    }

    // Phase 2: load bundle data if present
    rd->has_bundles = false;
    if (rd->header.bundle_index_offset != 0) {
        fp = fopen(path, "rb");
        if (fp) {
            fseek(fp, (long)rd->header.bundle_index_offset, SEEK_SET);
            if (fread(&rd->bundle_header, sizeof(jinf_nvmw_bundle_header), 1, fp) == 1) {
                rd->bundle_layer_descs.resize(rd->bundle_header.n_bundle_layers);
                bool ok = true;
                for (uint32_t i = 0; i < rd->bundle_header.n_bundle_layers; i++) {
                    if (fread(&rd->bundle_layer_descs[i], sizeof(jinf_nvmw_bundle_layer_desc), 1, fp) != 1) {
                        ok = false;
                        break;
                    }
                }
                rd->has_bundles = ok;
            }
            fclose(fp);
        }
    }

    *r = rd;
    return JINF_OK;
}

void jinf_nvmw_close(jinf_nvmw_reader* r) {
    delete r;
}

const jinf_nvmw_header* jinf_nvmw_get_header(const jinf_nvmw_reader* r) {
    return r ? &r->header : nullptr;
}

const jinf_nvmw_layer_desc* jinf_nvmw_get_layer_desc(const jinf_nvmw_reader* r, int layer) {
    if (!r || layer < 0 || layer >= (int)r->layer_descs.size()) return nullptr;
    return &r->layer_descs[layer];
}

const jinf_nvmw_tensor_entry* jinf_nvmw_get_tensor_entries(const jinf_nvmw_reader* r, int* count) {
    if (!r) { if (count) *count = 0; return nullptr; }
    if (count) *count = (int)r->tensor_entries.size();
    return r->tensor_entries.data();
}

jinf_status jinf_nvmw_get_layer_cold_range(const jinf_nvmw_reader* r, int layer,
                                            uint64_t* offset, uint64_t* size) {
    if (!r || !offset || !size) return JINF_ERR_INVALID;
    const jinf_nvmw_layer_desc* ld = jinf_nvmw_get_layer_desc(r, layer);
    if (!ld) return JINF_ERR_INVALID;
    *offset = ld->cold_start_offset;
    *size = ld->cold_total_size;
    return JINF_OK;
}

jinf_status jinf_nvmw_get_hot_range(const jinf_nvmw_reader* r, uint64_t* offset, uint64_t* size) {
    if (!r || !offset || !size) return JINF_ERR_INVALID;
    *offset = r->header.hot_offset;
    *size = r->header.hot_size;
    return JINF_OK;
}

const jinf_nvmw_tensor_entry* jinf_nvmw_find_tensor(const jinf_nvmw_reader* r, const char* name) {
    if (!r || !name) return nullptr;
    for (auto& te : r->tensor_entries) {
        if (strcmp(te.name, name) == 0) return &te;
    }
    return nullptr;
}

// ---- Bundle reader API (Phase 2) ----

bool jinf_nvmw_has_bundles(const jinf_nvmw_reader* r) {
    return r && r->has_bundles;
}

const jinf_nvmw_bundle_header* jinf_nvmw_get_bundle_header(const jinf_nvmw_reader* r) {
    if (!r || !r->has_bundles) return nullptr;
    return &r->bundle_header;
}

const jinf_nvmw_bundle_layer_desc* jinf_nvmw_get_bundle_layer_desc(const jinf_nvmw_reader* r, int layer) {
    if (!r || !r->has_bundles) return nullptr;
    if (layer < 0 || layer >= (int)r->bundle_layer_descs.size()) return nullptr;
    return &r->bundle_layer_descs[layer];
}

uint64_t jinf_nvmw_get_neuron_bundle_offset(const jinf_nvmw_reader* r, int layer, int neuron_id) {
    const jinf_nvmw_bundle_layer_desc* bld = jinf_nvmw_get_bundle_layer_desc(r, layer);
    if (!bld) return 0;
    return bld->bundles_offset + (uint64_t)neuron_id * bld->bundle_size;
}

// ---- Writer bundle API (Phase 2) ----

jinf_status jinf_nvmw_writer_write_bundle_section(jinf_nvmw_writer* w,
    const jinf_nvmw_bundle_header* bh,
    const jinf_nvmw_bundle_layer_desc* layer_descs,
    int n_layers) {
    if (!w || !w->fp || !bh || !layer_descs) return JINF_ERR_INVALID;

    // Pad to 4K before writing bundle index
    jinf_status s = pad_to_4k(w->fp);
    if (s != JINF_OK) return s;

    long bundle_index_pos = ftell(w->fp);

    // Write bundle header
    if (fwrite(bh, sizeof(jinf_nvmw_bundle_header), 1, w->fp) != 1) return JINF_ERR_IO;

    // Write per-layer bundle descriptors
    for (int i = 0; i < n_layers; i++) {
        if (fwrite(&layer_descs[i], sizeof(jinf_nvmw_bundle_layer_desc), 1, w->fp) != 1) return JINF_ERR_IO;
    }

    // Store the bundle index offset in the header
    w->header.bundle_index_offset = (uint64_t)bundle_index_pos;

    return JINF_OK;
}
