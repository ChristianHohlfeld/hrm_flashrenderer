#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/4] Build HRM core (Release)"
cmake -S "$ROOT/hrm_core" -B "$ROOT/hrm_core/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/hrm_core/build" -j

echo "[2/4] Run HRM unit tests"
ctest --test-dir "$ROOT/hrm_core/build"

echo "[3/4] Install Python package + CUDA extension (SM75)"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"
python -m pip install -U torch transformers safetensors
python -m pip install -e "$ROOT"

echo "[4/4] Flash kernel tests"
flash-kernel-test
flash-append-test

echo "OK"
