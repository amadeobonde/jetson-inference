#!/bin/bash
# benchmark_nvme.sh — Run NVMe throughput benchmarks on Jetson.
# Usage: ./scripts/benchmark_nvme.sh [test_file_path]
#
# Creates a test file if needed and runs nvme_bench with various configs.

set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
TEST_FILE="${1:-/tmp/nvme_bench_test.bin}"
TEST_SIZE_MB=2048

echo "=== NVMe Benchmark Suite ==="
echo "Build dir: ${BUILD_DIR}"
echo "Test file: ${TEST_FILE}"
echo ""

# Create test file if it doesn't exist
if [ ! -f "${TEST_FILE}" ]; then
    echo "Creating ${TEST_SIZE_MB}MB test file..."
    dd if=/dev/urandom of="${TEST_FILE}" bs=1M count=${TEST_SIZE_MB} status=progress 2>&1
    echo ""
fi

BENCH="${BUILD_DIR}/bench/nvme_bench"
if [ ! -x "${BENCH}" ]; then
    echo "Error: ${BENCH} not found. Build the project first:"
    echo "  mkdir -p build && cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

echo "--- Standard benchmark (1MB chunks, QD=64) ---"
${BENCH} "${TEST_FILE}" --size 1024 --chunk 1024 --depth 64
echo ""

echo "--- Large chunks (4MB) ---"
${BENCH} "${TEST_FILE}" --size 1024 --chunk 4096 --depth 32
echo ""

echo "--- Small chunks (256KB, high QD) ---"
${BENCH} "${TEST_FILE}" --size 512 --chunk 256 --depth 64
echo ""

echo "=== Done ==="
