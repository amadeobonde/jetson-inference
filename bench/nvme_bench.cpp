// nvme_bench — Raw NVMe throughput benchmark using io_uring + O_DIRECT.
// Usage: ./nvme_bench <file_path> [--size <MB>] [--depth <queue_depth>] [--chunk <KB>]

#include "jinf/nvme_io.h"
#include "jinf/common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

struct bench_config {
    const char* path;
    size_t total_mb;
    int queue_depth;
    size_t chunk_kb;
};

static bool parse_args(int argc, char** argv, bench_config* cfg) {
    cfg->path = nullptr;
    cfg->total_mb = 1024;
    cfg->queue_depth = 64;
    cfg->chunk_kb = 1024;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            cfg->total_mb = (size_t)atol(argv[++i]);
        } else if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) {
            cfg->queue_depth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--chunk") == 0 && i + 1 < argc) {
            cfg->chunk_kb = (size_t)atol(argv[++i]);
        } else if (argv[i][0] != '-') {
            cfg->path = argv[i];
        }
    }

    if (!cfg->path) {
        fprintf(stderr, "Usage: %s <file_path> [--size <MB>] [--depth <N>] [--chunk <KB>]\n", argv[0]);
        return false;
    }
    return true;
}

static double run_sequential_bench(const char* path, size_t chunk_size, size_t total_bytes,
                                    int queue_depth) {
    jinf_io_config io_cfg = {
        .queue_depth = queue_depth,
        .read_size = chunk_size,
        .flags = 0,
    };

    jinf_io_context* ctx = nullptr;
    if (jinf_io_create(&ctx, &io_cfg) != JINF_OK) {
        fprintf(stderr, "Failed to create I/O context\n");
        return -1.0;
    }

    int fd;
    if (jinf_io_open(ctx, path, &fd) != JINF_OK) {
        fprintf(stderr, "Failed to open %s\n", path);
        jinf_io_destroy(ctx);
        return -1.0;
    }

    void* buf = jinf_io_alloc_aligned(chunk_size);
    if (!buf) {
        fprintf(stderr, "Failed to allocate aligned buffer\n");
        jinf_io_destroy(ctx);
        return -1.0;
    }

    size_t bytes_read = 0;
    jinf_timer timer;

    while (bytes_read < total_bytes) {
        size_t remaining = total_bytes - bytes_read;
        size_t this_read = remaining < chunk_size ? JINF_ALIGN_4K(remaining) : chunk_size;

        jinf_status s = jinf_io_read_sync(ctx, fd, buf, bytes_read, this_read);
        if (s != JINF_OK) {
            fprintf(stderr, "Read failed at offset %zu\n", bytes_read);
            break;
        }
        bytes_read += this_read;
    }

    double elapsed_ms = timer.elapsed_ms();
    double gb_sec = (bytes_read / (1024.0 * 1024.0 * 1024.0)) / (elapsed_ms / 1000.0);

    jinf_io_free_aligned(buf);
    jinf_io_destroy(ctx);

    return gb_sec;
}

static double run_async_bench(const char* path, size_t chunk_size, size_t total_bytes,
                               int queue_depth) {
    jinf_io_config io_cfg = {
        .queue_depth = queue_depth,
        .read_size = chunk_size,
        .flags = 0,
    };

    jinf_io_context* ctx = nullptr;
    if (jinf_io_create(&ctx, &io_cfg) != JINF_OK) return -1.0;

    int fd;
    if (jinf_io_open(ctx, path, &fd) != JINF_OK) {
        jinf_io_destroy(ctx);
        return -1.0;
    }

    // Allocate multiple buffers for async I/O
    int n_buffers = queue_depth;
    void** bufs = (void**)malloc(n_buffers * sizeof(void*));
    for (int i = 0; i < n_buffers; i++) {
        bufs[i] = jinf_io_alloc_aligned(chunk_size);
    }

    size_t bytes_submitted = 0;
    size_t bytes_completed = 0;
    int in_flight = 0;
    jinf_timer timer;

    // Fill the queue
    while (bytes_submitted < total_bytes && in_flight < queue_depth) {
        int buf_idx = in_flight;
        size_t remaining = total_bytes - bytes_submitted;
        size_t this_read = remaining < chunk_size ? JINF_ALIGN_4K(remaining) : chunk_size;

        jinf_io_request req = {
            .fd = fd,
            .buffer = bufs[buf_idx],
            .offset = bytes_submitted,
            .length = this_read,
            .user_data = (void*)(intptr_t)buf_idx,
        };

        jinf_io_submit(ctx, &req, 1);
        bytes_submitted += this_read;
        in_flight++;
    }

    // Drain completions and submit more
    while (bytes_completed < total_bytes) {
        jinf_io_completion comps[64];
        int actual = 0;
        jinf_io_wait(ctx, comps, 1, 64, &actual);

        for (int i = 0; i < actual; i++) {
            if (comps[i].result < 0) {
                fprintf(stderr, "Async read failed: %d\n", comps[i].result);
                continue;
            }
            bytes_completed += comps[i].result;
            in_flight--;

            // Submit next read if more data remains
            if (bytes_submitted < total_bytes) {
                int buf_idx = (int)(intptr_t)comps[i].user_data;
                size_t remaining = total_bytes - bytes_submitted;
                size_t this_read = remaining < chunk_size ? JINF_ALIGN_4K(remaining) : chunk_size;

                jinf_io_request req = {
                    .fd = fd,
                    .buffer = bufs[buf_idx],
                    .offset = bytes_submitted,
                    .length = this_read,
                    .user_data = (void*)(intptr_t)buf_idx,
                };

                jinf_io_submit(ctx, &req, 1);
                bytes_submitted += this_read;
                in_flight++;
            }
        }
    }

    double elapsed_ms = timer.elapsed_ms();
    double gb_sec = (bytes_completed / (1024.0 * 1024.0 * 1024.0)) / (elapsed_ms / 1000.0);

    for (int i = 0; i < n_buffers; i++) {
        jinf_io_free_aligned(bufs[i]);
    }
    free(bufs);
    jinf_io_destroy(ctx);

    return gb_sec;
}

int main(int argc, char** argv) {
    bench_config cfg;
    if (!parse_args(argc, argv, &cfg)) return 1;

    size_t total_bytes = cfg.total_mb * 1024 * 1024;
    size_t chunk_bytes = cfg.chunk_kb * 1024;

    printf("=== NVMe Benchmark ===\n");
    printf("File:        %s\n", cfg.path);
    printf("Total read:  %zu MB\n", cfg.total_mb);
    printf("Chunk size:  %zu KB\n", cfg.chunk_kb);
    printf("Queue depth: %d\n\n", cfg.queue_depth);

    // Sequential (sync) benchmark
    printf("--- Sequential (sync) ---\n");
    double seq_gbps = run_sequential_bench(cfg.path, chunk_bytes, total_bytes, cfg.queue_depth);
    if (seq_gbps > 0) {
        printf("  Throughput: %.2f GB/s\n\n", seq_gbps);
    }

    // Async benchmark with varying queue depths
    int depths[] = {1, 8, 32, 64};
    printf("--- Async (io_uring) ---\n");
    for (int d : depths) {
        if (d > cfg.queue_depth) d = cfg.queue_depth;
        double gbps = run_async_bench(cfg.path, chunk_bytes, total_bytes, d);
        if (gbps > 0) {
            printf("  QD=%-3d  %.2f GB/s\n", d, gbps);
        }
    }

    // Varying chunk sizes
    size_t chunks[] = {256 * 1024, 512 * 1024, 1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024};
    printf("\n--- Chunk size sweep (QD=%d) ---\n", cfg.queue_depth);
    for (size_t c : chunks) {
        double gbps = run_async_bench(cfg.path, c, total_bytes, cfg.queue_depth);
        if (gbps > 0) {
            printf("  %4zu KB  %.2f GB/s\n", c / 1024, gbps);
        }
    }

    return 0;
}
