// prepare_model — Convert GGUF model to .nvmw format for NVMe streaming.
// Usage: ./prepare_model --input model.gguf --output model.nvmw --gpu-budget 4500

#include "jinf/gguf.h"
#include "jinf/nvmw.h"
#include "jinf/splitter.h"
#include "jinf/quant.h"
#include "jinf/common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

struct cli_args {
    const char* input;
    const char* output;
    size_t gpu_budget_mb;
    bool bundled;
};

static bool parse_args(int argc, char** argv, cli_args* args) {
    args->input = nullptr;
    args->output = nullptr;
    args->gpu_budget_mb = 4500;
    args->bundled = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            args->input = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            args->output = argv[++i];
        } else if (strcmp(argv[i], "--gpu-budget") == 0 && i + 1 < argc) {
            args->gpu_budget_mb = (size_t)atol(argv[++i]);
        } else if (strcmp(argv[i], "--bundled") == 0) {
            args->bundled = true;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return false;
        }
    }

    if (!args->input || !args->output) {
        fprintf(stderr, "Usage: %s --input <model.gguf> --output <model.nvmw> [--gpu-budget <MB>] [--bundled]\n",
                argv[0]);
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    cli_args args;
    if (!parse_args(argc, argv, &args)) return 1;

    // Parse GGUF
    printf("Parsing GGUF: %s\n", args.input);
    jinf_gguf_file* gguf = nullptr;
    jinf_status s = jinf_gguf_open(&gguf, args.input);
    if (s != JINF_OK) {
        fprintf(stderr, "Failed to parse GGUF: %s\n", jinf_status_str(s));
        return 1;
    }

    // Run splitter
    printf("Analyzing weights (GPU budget: %zu MB)...\n", args.gpu_budget_mb);
    jinf_split_config split_cfg = {
        .gpu_memory_budget = args.gpu_budget_mb * 1024 * 1024,
        .kv_cache_budget = 0,
        .buffer_budget = 0,
    };

    jinf_split_plan* plan = nullptr;
    s = jinf_split_analyze(gguf, &split_cfg, &plan);
    if (s != JINF_OK) {
        fprintf(stderr, "Split analysis failed: %s\n", jinf_status_str(s));
        jinf_gguf_close(gguf);
        return 1;
    }

    jinf_split_plan_print(plan);

    // Open GGUF for raw tensor data reads
    FILE* gguf_fp = fopen(args.input, "rb");
    if (!gguf_fp) {
        fprintf(stderr, "Failed to reopen GGUF for data reading\n");
        jinf_split_plan_free(plan);
        jinf_gguf_close(gguf);
        return 1;
    }

    // Create NVMW writer
    jinf_nvmw_writer* writer = nullptr;
    s = jinf_nvmw_writer_create(&writer, args.output);
    if (s != JINF_OK) {
        fprintf(stderr, "Failed to create NVMW writer: %s\n", jinf_status_str(s));
        fclose(gguf_fp);
        jinf_split_plan_free(plan);
        jinf_gguf_close(gguf);
        return 1;
    }

    // Build header
    jinf_nvmw_header hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = JINF_NVMW_MAGIC;
    hdr.version = JINF_NVMW_VERSION;
    hdr.n_layers = gguf->hparams.n_layers;
    hdr.n_tensors = (uint32_t)gguf->tensor_count;
    hdr.n_embd = gguf->hparams.n_embd;
    hdr.n_heads = gguf->hparams.n_heads;
    hdr.n_heads_kv = gguf->hparams.n_heads_kv;
    hdr.n_ff = gguf->hparams.n_ff;
    hdr.n_vocab = gguf->hparams.n_vocab;
    hdr.n_ctx_train = gguf->hparams.n_ctx_train;
    hdr.rope_freq_base = gguf->hparams.rope_freq_base;
    hdr.rms_norm_eps = gguf->hparams.rms_norm_eps;

    // Determine primary quant type from first large tensor
    for (uint64_t i = 0; i < gguf->tensor_count; i++) {
        if (gguf->tensor_infos[i].n_bytes > 1024 * 1024) {
            hdr.primary_type = (int32_t)gguf->tensor_infos[i].type;
            break;
        }
    }

    // Collect hot and cold tensors
    struct tensor_write {
        const jinf_gguf_tensor_info* info;
        int plan_index;
    };

    std::vector<tensor_write> hot_tensors;
    std::vector<std::vector<tensor_write>> cold_layers(hdr.n_layers);

    for (int i = 0; i < plan->count; i++) {
        const jinf_weight_assignment* a = &plan->assignments[i];
        tensor_write tw = { a->tensor, i };

        if (a->placement == JINF_PLACE_HOT) {
            hot_tensors.push_back(tw);
        } else if (a->layer_index >= 0 && a->layer_index < (int)hdr.n_layers) {
            cold_layers[a->layer_index].push_back(tw);
        }
    }

    // Compute sizes and offsets
    size_t hot_total = 0;
    for (auto& tw : hot_tensors) {
        hot_total += JINF_ALIGN_4K(tw.info->n_bytes);
    }

    // Index region: layer_descs + tensor_entries
    size_t index_size = hdr.n_layers * sizeof(jinf_nvmw_layer_desc)
                      + hdr.n_tensors * sizeof(jinf_nvmw_tensor_entry);

    hdr.index_offset = 4096;
    hdr.index_size = index_size;
    hdr.hot_offset = JINF_ALIGN_4K(4096 + index_size);
    hdr.hot_size = hot_total;
    hdr.cold_start_offset = JINF_ALIGN_4K(hdr.hot_offset + hot_total);

    // Build tensor entries and layer descs
    size_t hot_cursor = 0;
    for (auto& tw : hot_tensors) {
        jinf_nvmw_tensor_entry entry;
        memset(&entry, 0, sizeof(entry));
        strncpy(entry.name, tw.info->name, sizeof(entry.name) - 1);
        entry.type = tw.info->type;
        memcpy(entry.shape, tw.info->shape, sizeof(entry.shape));
        entry.n_dims = tw.info->n_dims;
        entry.offset = hot_cursor;
        entry.n_bytes = tw.info->n_bytes;
        entry.layer_index = plan->assignments[tw.plan_index].layer_index;
        entry.is_hot = 1;

        jinf_nvmw_writer_add_tensor_entry(writer, &entry);
        hot_cursor += JINF_ALIGN_4K(tw.info->n_bytes);
    }

    size_t cold_cursor = hdr.cold_start_offset;
    for (int layer = 0; layer < (int)hdr.n_layers; layer++) {
        size_t layer_cold_start = JINF_ALIGN_4K(cold_cursor);
        size_t layer_offset = 0;

        for (auto& tw : cold_layers[layer]) {
            jinf_nvmw_tensor_entry entry;
            memset(&entry, 0, sizeof(entry));
            strncpy(entry.name, tw.info->name, sizeof(entry.name) - 1);
            entry.type = tw.info->type;
            memcpy(entry.shape, tw.info->shape, sizeof(entry.shape));
            entry.n_dims = tw.info->n_dims;
            entry.offset = layer_offset;  // relative to layer cold start
            entry.n_bytes = tw.info->n_bytes;
            entry.layer_index = layer;
            entry.is_hot = 0;

            jinf_nvmw_writer_add_tensor_entry(writer, &entry);
            layer_offset += JINF_ALIGN_4K(tw.info->n_bytes);
        }

        jinf_nvmw_layer_desc ld;
        memset(&ld, 0, sizeof(ld));
        ld.cold_start_offset = layer_cold_start;
        ld.cold_total_size = layer_offset;
        ld.n_cold_tensors = (uint32_t)cold_layers[layer].size();
        jinf_nvmw_writer_add_layer_desc(writer, layer, &ld);

        cold_cursor = layer_cold_start + layer_offset;
    }

    hdr.cold_total_size = cold_cursor - hdr.cold_start_offset;

    jinf_nvmw_writer_set_header(writer, &hdr);
    s = jinf_nvmw_writer_finalize(writer);
    if (s != JINF_OK) {
        fprintf(stderr, "Failed to finalize NVMW header: %s\n", jinf_status_str(s));
        fclose(gguf_fp);
        jinf_nvmw_writer_destroy(writer);
        jinf_split_plan_free(plan);
        jinf_gguf_close(gguf);
        return 1;
    }

    // Write hot tensor data
    printf("Writing hot region: %.1f MB\n", hot_total / (1024.0 * 1024.0));
    // Seek writer's file to hot_offset
    // The writer's finalize positioned us at hot_offset

    FILE* out_fp = fopen(args.output, "r+b");
    if (!out_fp) {
        fprintf(stderr, "Failed to reopen output for data writing\n");
        fclose(gguf_fp);
        jinf_nvmw_writer_destroy(writer);
        jinf_split_plan_free(plan);
        jinf_gguf_close(gguf);
        return 1;
    }

    fseek(out_fp, (long)hdr.hot_offset, SEEK_SET);

    // Read each hot tensor from GGUF and write to NVMW
    std::vector<uint8_t> copy_buf;
    for (auto& tw : hot_tensors) {
        size_t aligned = JINF_ALIGN_4K(tw.info->n_bytes);
        copy_buf.resize(aligned, 0);

        fseek(gguf_fp, (long)(gguf->data_offset + tw.info->offset), SEEK_SET);
        fread(copy_buf.data(), 1, tw.info->n_bytes, gguf_fp);

        fwrite(copy_buf.data(), 1, aligned, out_fp);
    }

    // Write cold tensor data
    size_t total_cold_written = 0;
    for (int layer = 0; layer < (int)hdr.n_layers; layer++) {
        if (cold_layers[layer].empty()) continue;

        const jinf_nvmw_layer_desc* ld = nullptr;
        // Use the desc we built
        jinf_nvmw_layer_desc local_ld;
        // Recompute offset (we already calculated it)
        // Actually, seek to the layer's cold_start_offset
        // We can get this from our calculated offsets
        // For simplicity, use sequential writing (the writer positioned us correctly)

        for (auto& tw : cold_layers[layer]) {
            size_t aligned = JINF_ALIGN_4K(tw.info->n_bytes);
            copy_buf.resize(aligned, 0);

            fseek(gguf_fp, (long)(gguf->data_offset + tw.info->offset), SEEK_SET);
            fread(copy_buf.data(), 1, tw.info->n_bytes, gguf_fp);

            // Pad to 4K
            memset(copy_buf.data() + tw.info->n_bytes, 0, aligned - tw.info->n_bytes);

            fwrite(copy_buf.data(), 1, aligned, out_fp);
            total_cold_written += aligned;
        }

        // Pad layer to 4K boundary
        long pos = ftell(out_fp);
        size_t padded = JINF_ALIGN_4K((size_t)pos);
        if (padded > (size_t)pos) {
            std::vector<uint8_t> pad(padded - pos, 0);
            fwrite(pad.data(), 1, pad.size(), out_fp);
        }
    }

    // Phase 2: Write neuron bundle section if --bundled
    size_t bundle_total = 0;
    if (args.bundled) {
        printf("\nWriting neuron bundles...\n");

        uint32_t n_embd = hdr.n_embd;
        uint32_t n_ff = hdr.n_ff;

        // Compute bundle size: for each neuron, gate_row + up_row + down_col(float32)
        // Gate and up are quantized rows, but down column is stored as float32
        // (extracting columns from block-quantized matrices requires dequantization).
        jinf_qtype qt = (jinf_qtype)hdr.primary_type;
        int block_size = jinf_qtype_block_size(qt);
        size_t type_size = jinf_qtype_type_size(qt);
        size_t row_bytes = (size_t)((n_embd + block_size - 1) / block_size) * type_size;
        size_t down_col_bytes = (size_t)n_embd * sizeof(float);
        size_t raw_bundle = 2 * row_bytes + down_col_bytes;
        size_t padded_bundle = JINF_ALIGN_4K(raw_bundle);

        printf("  Bundle: gate/up_row=%zu, down_col(f32)=%zu, raw=%zu, padded=%zu (per neuron)\n",
               row_bytes, down_col_bytes, raw_bundle, padded_bundle);

        // Prepare bundle layer descriptors
        std::vector<jinf_nvmw_bundle_layer_desc> bundle_layer_descs(hdr.n_layers);
        int n_bundle_layers = 0;

        // Pad output to 4K before bundles
        long cur_pos = ftell(out_fp);
        size_t padded_pos = JINF_ALIGN_4K((size_t)cur_pos);
        if (padded_pos > (size_t)cur_pos) {
            std::vector<uint8_t> pad(padded_pos - cur_pos, 0);
            fwrite(pad.data(), 1, pad.size(), out_fp);
        }

        // For each layer, repack FFN tensors into neuron bundles
        for (int layer = 0; layer < (int)hdr.n_layers; layer++) {
            // Find FFN gate, up, down tensor entries for this layer
            const jinf_gguf_tensor_info* gate_info = nullptr;
            const jinf_gguf_tensor_info* up_info = nullptr;
            const jinf_gguf_tensor_info* down_info = nullptr;

            for (uint64_t i = 0; i < gguf->tensor_count; i++) {
                const jinf_gguf_tensor_info* ti = &gguf->tensor_infos[i];
                char layer_prefix[64];
                snprintf(layer_prefix, sizeof(layer_prefix), "blk.%d.", layer);
                if (strstr(ti->name, layer_prefix) == nullptr) continue;

                if (strstr(ti->name, "ffn_gate.weight")) gate_info = ti;
                else if (strstr(ti->name, "ffn_up.weight")) up_info = ti;
                else if (strstr(ti->name, "ffn_down.weight")) down_info = ti;
            }

            if (!gate_info || !up_info || !down_info) {
                // No FFN tensors for this layer (shouldn't happen), skip
                memset(&bundle_layer_descs[layer], 0, sizeof(jinf_nvmw_bundle_layer_desc));
                continue;
            }

            long bundles_start = ftell(out_fp);
            bundle_layer_descs[layer].bundles_offset = (uint64_t)bundles_start;
            bundle_layer_descs[layer].n_neurons = n_ff;
            bundle_layer_descs[layer].bundle_size = (uint32_t)padded_bundle;

            // Compute actual row sizes per tensor (may differ from primary type)
            jinf_qtype gate_qt = gate_info->type;
            jinf_qtype up_qt = up_info->type;
            jinf_qtype down_qt = down_info->type;

            int gate_bs = jinf_qtype_block_size(gate_qt);
            int up_bs = jinf_qtype_block_size(up_qt);
            int down_bs = jinf_qtype_block_size(down_qt);

            size_t gate_row_bytes = (size_t)((n_embd + gate_bs - 1) / gate_bs) * jinf_qtype_type_size(gate_qt);
            size_t up_row_bytes = (size_t)((n_embd + up_bs - 1) / up_bs) * jinf_qtype_type_size(up_qt);
            size_t down_row_bytes = (size_t)((n_embd + down_bs - 1) / down_bs) * jinf_qtype_type_size(down_qt);

            // Read full gate, up, down matrices from GGUF
            std::vector<uint8_t> gate_data(gate_info->n_bytes);
            std::vector<uint8_t> up_data(up_info->n_bytes);
            std::vector<uint8_t> down_data(down_info->n_bytes);

            fseek(gguf_fp, (long)(gguf->data_offset + gate_info->offset), SEEK_SET);
            fread(gate_data.data(), 1, gate_info->n_bytes, gguf_fp);
            fseek(gguf_fp, (long)(gguf->data_offset + up_info->offset), SEEK_SET);
            fread(up_data.data(), 1, up_info->n_bytes, gguf_fp);
            fseek(gguf_fp, (long)(gguf->data_offset + down_info->offset), SEEK_SET);
            fread(down_data.data(), 1, down_info->n_bytes, gguf_fp);

            // Write bundles: for each neuron i, pack gate_row[i] + up_row[i] + down_col[i]
            // gate and up are [n_ff x n_embd], row i = bytes [i*row_bytes .. (i+1)*row_bytes)
            // down is [n_embd x n_ff], column i = row i of transposed = same layout as row-major row i
            // In GGUF, weights are stored row-major as [out_features x in_features],
            // so down.weight is [n_embd x n_ff] and "column i" = extracting every n_ff-th element.
            // But for K-quant formats, the data is blocked, and extracting columns is non-trivial.
            // However, in ggml, down.weight is stored as [n_embd x n_ff] but the matvec does
            // output[row] = dot(weight_row, input), so input is [n_ff] and output is [n_embd].
            // That means weight is [n_embd x n_ff] row-major: row i has n_ff elements.
            // "Column j" of this = element j from each of the n_embd rows.
            //
            // For bundled format, we need the "column j" as a contiguous vector [n_embd].
            // With block quantization this requires re-blocking. For MVP, we dequantize the
            // full down matrix, extract columns as float, then re-quantize. This is slow but
            // only done once during model prep.
            //
            // Simpler approach: store down rows (not columns) in bundles, and adjust the sparse
            // kernel to do the transposed multiply. Actually, for the down projection the operation
            // is output[d] += activated_value * down_weight[d][neuron]. Since down is [n_embd x n_ff],
            // we need weight[d][neuron] for all d, which IS column `neuron` of the down matrix.
            //
            // For K-quant, we'll dequantize to float, extract column, and write as raw float.
            // The sparse kernel already handles this per-element.
            //
            // Actually, let's use a better approach: store down_col as Q8_0 re-quantized from float.
            // For MVP simplicity, store as raw float [n_embd * 4 bytes].
            // But that's 4096*4 = 16KB per neuron for down alone, making bundles much larger.
            //
            // Best approach: since down.weight is [n_embd x n_ff] and we want column j,
            // and in ggml row-major layout each row has n_ff elements,
            // column j = {row[0][j], row[1][j], ...row[n_embd-1][j]}
            // For Q4_K with block_size=256: row[r] is (n_ff/256) blocks.
            // Element j in row r: block_idx = j/256, elem_in_block = j%256.
            //
            // For the bundle, we want to pack the "column slice" — but this can't be a
            // contiguous block-quantized vector without re-encoding. So instead, we store
            // the down projection differently: we transpose the down matrix row-by-row into
            // [n_ff x n_embd] format (same as gate/up), then row i of transposed = column i
            // of original. We do this by dequantizing to float and re-quantizing per-row.

            // Dequantize full down matrix to float for column extraction
            int64_t down_n_elements = (int64_t)n_embd * n_ff;
            std::vector<float> down_float(down_n_elements);
            jinf_dequantize(down_data.data(), down_qt, down_float.data(), down_n_elements);

            // Now down_float is [n_embd][n_ff] row-major
            // Column j: down_float[r * n_ff + j] for r = 0..n_embd-1

            // Prepare a buffer for one bundle (padded)
            std::vector<uint8_t> bundle_buf(padded_bundle, 0);

            for (uint32_t neuron = 0; neuron < n_ff; neuron++) {
                memset(bundle_buf.data(), 0, padded_bundle);

                // Copy gate row i
                memcpy(bundle_buf.data(),
                       gate_data.data() + neuron * gate_row_bytes,
                       gate_row_bytes);

                // Copy up row i
                memcpy(bundle_buf.data() + gate_row_bytes,
                       up_data.data() + neuron * up_row_bytes,
                       up_row_bytes);

                // For down column: extract and store as float32 [n_embd]
                // Offset within bundle: after gate + up rows
                float* down_col_ptr = (float*)(bundle_buf.data() + gate_row_bytes + up_row_bytes);
                for (uint32_t d = 0; d < n_embd; d++) {
                    down_col_ptr[d] = down_float[(size_t)d * n_ff + neuron];
                }

                fwrite(bundle_buf.data(), 1, padded_bundle, out_fp);
                bundle_total += padded_bundle;
            }

            bundle_layer_descs[layer].bundles_total_size = (uint64_t)n_ff * padded_bundle;
            n_bundle_layers++;

            if ((layer + 1) % 8 == 0) {
                printf("  Bundled layers: %d/%d\n", layer + 1, (int)hdr.n_layers);
            }
        }

        // Write bundle index section
        jinf_nvmw_bundle_header bh;
        memset(&bh, 0, sizeof(bh));
        bh.n_bundle_layers = hdr.n_layers;
        bh.bundle_size = (uint32_t)padded_bundle;
        bh.n_ff = n_ff;

        // Pad to 4K before bundle index
        cur_pos = ftell(out_fp);
        padded_pos = JINF_ALIGN_4K((size_t)cur_pos);
        if (padded_pos > (size_t)cur_pos) {
            std::vector<uint8_t> pad(padded_pos - cur_pos, 0);
            fwrite(pad.data(), 1, pad.size(), out_fp);
        }

        long bundle_index_offset = ftell(out_fp);

        fwrite(&bh, sizeof(bh), 1, out_fp);
        for (uint32_t i = 0; i < hdr.n_layers; i++) {
            fwrite(&bundle_layer_descs[i], sizeof(jinf_nvmw_bundle_layer_desc), 1, out_fp);
        }

        // Update header with bundle_index_offset and version
        hdr.bundle_index_offset = (uint64_t)bundle_index_offset;
        hdr.version = JINF_NVMW_VERSION_BUNDLED;
        fseek(out_fp, 0, SEEK_SET);
        fwrite(&hdr, sizeof(hdr), 1, out_fp);

        printf("  Bundle index at offset: %ld\n", bundle_index_offset);
        printf("  Total bundle data: %.1f MB\n", bundle_total / (1024.0 * 1024.0));
    }

    fclose(out_fp);
    fclose(gguf_fp);

    printf("\nConversion complete!\n");
    printf("  Hot:  %.1f MB (%zu tensors)\n",
           hot_total / (1024.0 * 1024.0), hot_tensors.size());
    printf("  Cold: %.1f MB (%zu layers)\n",
           total_cold_written / (1024.0 * 1024.0), (size_t)hdr.n_layers);
    if (args.bundled) {
        printf("  Bundles: %.1f MB\n", bundle_total / (1024.0 * 1024.0));
    }
    printf("  Output: %s\n", args.output);

    jinf_nvmw_writer_destroy(writer);
    jinf_split_plan_free(plan);
    jinf_gguf_close(gguf);
    return 0;
}
