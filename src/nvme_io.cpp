#include "jinf/nvme_io.h"

#include <liburing.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <vector>

// ---- Internal context ----

struct jinf_io_context {
    struct io_uring ring;
    int    queue_depth;
    size_t read_size;
    int    pending;           // number of submitted but not yet completed requests
    std::vector<int> open_fds;
};

// ---- Aligned allocation ----

void* jinf_io_alloc_aligned(size_t size) {
    size = JINF_ALIGN_4K(size);
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, size) != 0) {
        return nullptr;
    }
    return ptr;
}

void jinf_io_free_aligned(void* ptr) {
    free(ptr);
}

// ---- Create / Destroy ----

jinf_status jinf_io_create(jinf_io_context** ctx, const jinf_io_config* config) {
    if (!ctx || !config) return JINF_ERR_INVALID;

    jinf_io_context* c = new (std::nothrow) jinf_io_context();
    if (!c) return JINF_ERR_OOM;

    c->queue_depth = config->queue_depth > 0 ? config->queue_depth : 64;
    c->read_size = config->read_size > 0 ? config->read_size : (1024 * 1024);
    c->pending = 0;

    int ret = io_uring_queue_init(c->queue_depth, &c->ring, 0);
    if (ret < 0) {
        JINF_LOG("io_uring_queue_init failed: %s", strerror(-ret));
        delete c;
        return JINF_ERR_IO;
    }

    *ctx = c;
    return JINF_OK;
}

void jinf_io_destroy(jinf_io_context* ctx) {
    if (!ctx) return;
    for (int fd : ctx->open_fds) {
        close(fd);
    }
    io_uring_queue_exit(&ctx->ring);
    delete ctx;
}

// ---- Open / Close ----

jinf_status jinf_io_open(jinf_io_context* ctx, const char* path, int* fd) {
    if (!ctx || !path || !fd) return JINF_ERR_INVALID;

    int f = open(path, O_RDONLY | O_DIRECT);
    if (f < 0) {
        JINF_LOG("open(%s) failed: %s", path, strerror(errno));
        return JINF_ERR_IO;
    }

    ctx->open_fds.push_back(f);
    *fd = f;
    return JINF_OK;
}

void jinf_io_close(jinf_io_context* ctx, int fd) {
    if (!ctx) return;
    close(fd);
    for (auto it = ctx->open_fds.begin(); it != ctx->open_fds.end(); ++it) {
        if (*it == fd) {
            ctx->open_fds.erase(it);
            break;
        }
    }
}

// ---- Submit ----

jinf_status jinf_io_submit(jinf_io_context* ctx, const jinf_io_request* reqs, int count) {
    if (!ctx || !reqs || count <= 0) return JINF_ERR_INVALID;

    for (int i = 0; i < count; i++) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ctx->ring);
        if (!sqe) {
            JINF_LOG("io_uring_get_sqe failed (queue full, pending=%d)", ctx->pending);
            return JINF_ERR_IO;
        }

        // Use a single iovec for each request
        // We store the iovec in the user_data area by wrapping the request
        struct iovec iov;
        iov.iov_base = reqs[i].buffer;
        iov.iov_len = reqs[i].length;

        io_uring_prep_readv(sqe, reqs[i].fd, &iov, 1, reqs[i].offset);
        io_uring_sqe_set_data(sqe, reqs[i].user_data);
    }

    int submitted = io_uring_submit(&ctx->ring);
    if (submitted < 0) {
        JINF_LOG("io_uring_submit failed: %s", strerror(-submitted));
        return JINF_ERR_IO;
    }
    if (submitted != count) {
        JINF_LOG("io_uring_submit: only %d of %d submitted", submitted, count);
    }

    ctx->pending += submitted;
    return JINF_OK;
}

// ---- Wait ----

jinf_status jinf_io_wait(jinf_io_context* ctx, jinf_io_completion* comps, int min, int max, int* actual) {
    if (!ctx || !comps || !actual) return JINF_ERR_INVALID;
    if (min <= 0) min = 1;
    if (max <= 0) max = min;

    // Wait for at least `min` completions
    struct io_uring_cqe* cqe;
    int ret = io_uring_wait_cqe_nr(&ctx->ring, &cqe, min);
    if (ret < 0) {
        JINF_LOG("io_uring_wait_cqe_nr failed: %s", strerror(-ret));
        return JINF_ERR_IO;
    }

    // Peek all available completions (up to max)
    struct io_uring_cqe* cqes[256];
    int ready = max > 256 ? 256 : max;
    unsigned count = io_uring_peek_batch_cqe(&ctx->ring, cqes, ready);

    int n = (int)count;
    for (int i = 0; i < n; i++) {
        comps[i].user_data = io_uring_cqe_get_data(cqes[i]);
        comps[i].result = cqes[i]->res;
        io_uring_cqe_seen(&ctx->ring, cqes[i]);
    }

    ctx->pending -= n;
    *actual = n;
    return JINF_OK;
}

// ---- Synchronous read ----

jinf_status jinf_io_read_sync(jinf_io_context* ctx, int fd, void* buf, size_t offset, size_t length) {
    if (!ctx || !buf) return JINF_ERR_INVALID;

    // Ensure alignment for O_DIRECT
    size_t aligned_len = JINF_ALIGN_4K(length);

    jinf_io_request req = {
        .fd = fd,
        .buffer = buf,
        .offset = offset,
        .length = aligned_len,
        .user_data = nullptr,
    };

    jinf_status s = jinf_io_submit(ctx, &req, 1);
    if (s != JINF_OK) return s;

    jinf_io_completion comp;
    int actual = 0;
    s = jinf_io_wait(ctx, &comp, 1, 1, &actual);
    if (s != JINF_OK) return s;

    if (comp.result < 0) {
        JINF_LOG("read_sync: I/O error %d", comp.result);
        return JINF_ERR_IO;
    }

    return JINF_OK;
}
