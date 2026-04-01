#!/usr/bin/env bash
set -euo pipefail

if command -v cmake >/dev/null 2>&1; then
  echo "[cpp] hrm_core build + tests"
  cmake -S hrm_core -B hrm_core/build -DCMAKE_BUILD_TYPE=Release
  cmake --build hrm_core/build -j
  ctest --test-dir hrm_core/build
else
  echo "[skip] cmake not found; skipping hrm_core build/tests"
fi

echo "[python] hrm_api test"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERR: python3/python not found; cannot run python tests" >&2
  exit 1
fi
"$PYTHON_BIN" -m tests.test_hrm_api

echo "[bash] topology detection test"
bash scripts/test_topology_detection.sh

if command -v flash-kernel-test >/dev/null 2>&1; then
  echo "[cuda] flash kernel tests"
  flash-kernel-test
  flash-append-test
else
  echo "[skip] flash-kernel-test not found; skipping flash kernel tests"
fi
