#!/bin/bash
# benchmark_inference.sh — Run inference benchmarks.
# Usage: ./scripts/benchmark_inference.sh <model.nvmw> [options]

set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
MODEL_PATH="${1:?Usage: $0 <model.nvmw> [--tokens N] [--gpu-budget MB]}"
shift

BENCH="${BUILD_DIR}/bench/inference_bench"
if [ ! -x "${BENCH}" ]; then
    echo "Error: ${BENCH} not found. Build the project first."
    exit 1
fi

echo "=== Inference Benchmark ==="
echo "Model: ${MODEL_PATH}"
echo ""

# Short benchmark (16 tokens)
echo "--- Short generation (16 tokens) ---"
${BENCH} "${MODEL_PATH}" --tokens 16 "$@"
echo ""

# Medium benchmark (64 tokens)
echo "--- Medium generation (64 tokens) ---"
${BENCH} "${MODEL_PATH}" --tokens 64 "$@"
echo ""

# Long benchmark (128 tokens)
echo "--- Long generation (128 tokens) ---"
${BENCH} "${MODEL_PATH}" --tokens 128 "$@"
echo ""

echo "=== Done ==="
