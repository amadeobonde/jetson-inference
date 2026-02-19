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
};

static bool parse_args(int argc, char** argv, cli_args* args) {
    args->input = nullptr;
    args->output = nullptr;
    args->gpu_budget_mb = 4500;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            args->input = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            args->output = argv[++i];
        } else if (strcmp(argv[i], "--gpu-budget") == 0 && i + 1 < argc) {
            args->gpu_budget_mb = (size_t)atol(argv[++i]);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return false;
        }
    }

    if (!args->input || !args->output) {
        fprintf(stderr, "Usage: %s --input <model.gguf> --output <model.nvmw> [--gpu-budget <MB>]\n",
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

    fclose(out_fp);
    fclose(gguf_fp);

    printf("\nConversion complete!\n");
    printf("  Hot:  %.1f MB (%zu tensors)\n",
           hot_total / (1024.0 * 1024.0), hot_tensors.size());
    printf("  Cold: %.1f MB (%zu layers)\n",
           total_cold_written / (1024.0 * 1024.0), (size_t)hdr.n_layers);
    printf("  Output: %s\n", args.output);

    jinf_nvmw_writer_destroy(writer);
    jinf_split_plan_free(plan);
    jinf_gguf_close(gguf);
    return 0;
}
