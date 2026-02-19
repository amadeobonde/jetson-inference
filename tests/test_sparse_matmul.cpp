// test_sparse_matmul — Phase 2: Unit tests for sparse FFN kernels.
// Verifies correctness of sparse matmul vs dense reference.

#include "jinf/sparse_ops.h"
#include "jinf/common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>

static int test_placeholder() {
    printf("  test_placeholder...\n");

    // Phase 2 test placeholder.
    // Full test requires neuron-bundled data and CUDA device.
    // Verify the header compiles and basic types are correct.

    jinf_sparse_ffn_args args;
    memset(&args, 0, sizeof(args));
    assert(args.num_activated == 0);
    assert(args.hidden_dim == 0);

    printf("    PASSED (placeholder — Phase 2)\n");
    return 0;
}

int main() {
    printf("=== test_sparse_matmul (Phase 2) ===\n");

    test_placeholder();

    printf("Sparse matmul tests passed (placeholder)!\n");
    return 0;
}
