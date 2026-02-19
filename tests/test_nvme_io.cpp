// test_nvme_io — Unit tests for NVMe I/O layer.
// Tests io_uring submit/wait with a temp file.

#include "jinf/nvme_io.h"
#include "jinf/common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

static const char* TEST_FILE = "/tmp/test_jinf_io.bin";
static const size_t TEST_SIZE = 4 * 1024 * 1024;  // 4MB

static void create_test_file() {
    FILE* f = fopen(TEST_FILE, "wb");
    assert(f);

    // Write known pattern: each 4K page starts with its page number
    uint8_t page[4096];
    size_t n_pages = TEST_SIZE / 4096;
    for (size_t i = 0; i < n_pages; i++) {
        memset(page, 0, sizeof(page));
        uint32_t page_num = (uint32_t)i;
        memcpy(page, &page_num, sizeof(page_num));
        // Fill rest with deterministic data
        for (size_t j = 4; j < 4096; j++) {
            page[j] = (uint8_t)((i + j) & 0xFF);
        }
        fwrite(page, 1, 4096, f);
    }
    fclose(f);
}

static int test_sync_read() {
    printf("  test_sync_read...\n");

    jinf_io_config cfg = jinf_io_config_default();
    jinf_io_context* ctx = nullptr;
    assert(jinf_io_create(&ctx, &cfg) == JINF_OK);

    int fd;
    assert(jinf_io_open(ctx, TEST_FILE, &fd) == JINF_OK);

    void* buf = jinf_io_alloc_aligned(4096);
    assert(buf);

    // Read first page
    assert(jinf_io_read_sync(ctx, fd, buf, 0, 4096) == JINF_OK);

    uint32_t page_num;
    memcpy(&page_num, buf, sizeof(page_num));
    assert(page_num == 0);

    // Read page 10
    assert(jinf_io_read_sync(ctx, fd, buf, 10 * 4096, 4096) == JINF_OK);
    memcpy(&page_num, buf, sizeof(page_num));
    assert(page_num == 10);

    jinf_io_free_aligned(buf);
    jinf_io_destroy(ctx);
    printf("    PASSED\n");
    return 0;
}

static int test_async_read() {
    printf("  test_async_read...\n");

    jinf_io_config cfg = jinf_io_config_default();
    jinf_io_context* ctx = nullptr;
    assert(jinf_io_create(&ctx, &cfg) == JINF_OK);

    int fd;
    assert(jinf_io_open(ctx, TEST_FILE, &fd) == JINF_OK);

    // Submit 4 async reads of 4KB each
    void* bufs[4];
    jinf_io_request reqs[4];
    for (int i = 0; i < 4; i++) {
        bufs[i] = jinf_io_alloc_aligned(4096);
        assert(bufs[i]);
        reqs[i] = {
            .fd = fd,
            .buffer = bufs[i],
            .offset = (size_t)i * 4096,
            .length = 4096,
            .user_data = (void*)(intptr_t)i,
        };
    }

    assert(jinf_io_submit(ctx, reqs, 4) == JINF_OK);

    jinf_io_completion comps[4];
    int actual = 0;
    assert(jinf_io_wait(ctx, comps, 4, 4, &actual) == JINF_OK);
    assert(actual >= 1);  // At least 1 completion (may get all 4)

    // Verify data
    for (int i = 0; i < actual; i++) {
        assert(comps[i].result > 0);
    }

    // Check each buffer has correct page number
    for (int i = 0; i < 4; i++) {
        uint32_t page_num;
        memcpy(&page_num, bufs[i], sizeof(page_num));
        assert(page_num == (uint32_t)i);
        jinf_io_free_aligned(bufs[i]);
    }

    jinf_io_destroy(ctx);
    printf("    PASSED\n");
    return 0;
}

static int test_large_read() {
    printf("  test_large_read...\n");

    jinf_io_config cfg = jinf_io_config_default();
    jinf_io_context* ctx = nullptr;
    assert(jinf_io_create(&ctx, &cfg) == JINF_OK);

    int fd;
    assert(jinf_io_open(ctx, TEST_FILE, &fd) == JINF_OK);

    // Read 1MB at once
    size_t read_size = 1024 * 1024;
    void* buf = jinf_io_alloc_aligned(read_size);
    assert(buf);

    assert(jinf_io_read_sync(ctx, fd, buf, 0, read_size) == JINF_OK);

    // Verify first and last pages
    uint32_t page_num;
    memcpy(&page_num, buf, sizeof(page_num));
    assert(page_num == 0);

    memcpy(&page_num, (char*)buf + (255 * 4096), sizeof(page_num));
    assert(page_num == 255);

    jinf_io_free_aligned(buf);
    jinf_io_destroy(ctx);
    printf("    PASSED\n");
    return 0;
}

int main() {
    printf("=== test_nvme_io ===\n");
    create_test_file();

    test_sync_read();
    test_async_read();
    test_large_read();

    printf("All NVMe I/O tests passed!\n");
    return 0;
}
