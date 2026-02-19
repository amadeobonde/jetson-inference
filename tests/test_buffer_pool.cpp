// test_buffer_pool — Unit tests for double-buffer pipeline.
// Tests buffer creation, swap, and basic I/O pipeline.

#include "jinf/buffer_pool.h"
#include "jinf/nvme_io.h"
#include "jinf/common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

static const char* TEST_FILE = "/tmp/test_jinf_buffer.bin";
static const size_t TEST_FILE_SIZE = 2 * 1024 * 1024;  // 2MB

static void create_test_file() {
    FILE* f = fopen(TEST_FILE, "wb");
    assert(f);

    uint8_t page[4096];
    for (size_t i = 0; i < TEST_FILE_SIZE / 4096; i++) {
        memset(page, (uint8_t)(i & 0xFF), sizeof(page));
        uint32_t marker = (uint32_t)i;
        memcpy(page, &marker, sizeof(marker));
        fwrite(page, 1, 4096, f);
    }
    fclose(f);
}

static int test_buffer_create() {
    printf("  test_buffer_create...\n");

    jinf_buffer_config cfg = { .capacity = 1024 * 1024 };  // 1MB
    jinf_buffer_pair* bp = nullptr;

    assert(jinf_buffer_create(&bp, &cfg) == JINF_OK);
    assert(bp != nullptr);
    assert(bp->host_pinned[0] != nullptr);
    assert(bp->host_pinned[1] != nullptr);
    assert(bp->capacity == 1024 * 1024);
    assert(bp->active == 0);

    jinf_buffer_destroy(bp);
    printf("    PASSED\n");
    return 0;
}

static int test_buffer_swap() {
    printf("  test_buffer_swap...\n");

    jinf_buffer_config cfg = { .capacity = 4096 };
    jinf_buffer_pair* bp = nullptr;
    assert(jinf_buffer_create(&bp, &cfg) == JINF_OK);

    assert(bp->active == 0);
    void* ptr0 = jinf_buffer_active_ptr(bp);

    jinf_buffer_swap(bp);
    assert(bp->active == 1);
    void* ptr1 = jinf_buffer_active_ptr(bp);
    assert(ptr0 != ptr1);

    jinf_buffer_swap(bp);
    assert(bp->active == 0);
    assert(jinf_buffer_active_ptr(bp) == ptr0);

    jinf_buffer_destroy(bp);
    printf("    PASSED\n");
    return 0;
}

static int test_buffer_read_pipeline() {
    printf("  test_buffer_read_pipeline...\n");

    // Create I/O context
    jinf_io_config io_cfg = jinf_io_config_default();
    jinf_io_context* io = nullptr;
    assert(jinf_io_create(&io, &io_cfg) == JINF_OK);

    int fd;
    assert(jinf_io_open(io, TEST_FILE, &fd) == JINF_OK);

    // Create buffer pair
    jinf_buffer_config buf_cfg = { .capacity = 1024 * 1024 };
    jinf_buffer_pair* bp = nullptr;
    assert(jinf_buffer_create(&bp, &buf_cfg) == JINF_OK);

    // Start read into inactive buffer
    assert(jinf_buffer_start_read(bp, io, fd, 0, 4096) == JINF_OK);

    // Wait for completion
    void* ptr = nullptr;
    size_t size = 0;
    assert(jinf_buffer_wait_read(bp, io, &ptr, &size) == JINF_OK);
    assert(ptr != nullptr);
    assert(size == 4096);

    // Verify data (should be page 0)
    uint32_t marker;
    memcpy(&marker, ptr, sizeof(marker));
    assert(marker == 0);

    // Swap and read next chunk
    jinf_buffer_swap(bp);
    assert(jinf_buffer_start_read(bp, io, fd, 4096, 4096) == JINF_OK);
    assert(jinf_buffer_wait_read(bp, io, &ptr, &size) == JINF_OK);

    memcpy(&marker, ptr, sizeof(marker));
    assert(marker == 1);

    jinf_buffer_destroy(bp);
    jinf_io_destroy(io);
    printf("    PASSED\n");
    return 0;
}

int main() {
    printf("=== test_buffer_pool ===\n");
    create_test_file();

    test_buffer_create();
    test_buffer_swap();
    test_buffer_read_pipeline();

    printf("All buffer pool tests passed!\n");
    return 0;
}
