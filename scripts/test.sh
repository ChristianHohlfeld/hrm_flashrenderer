#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

if command -v cmake >/dev/null 2>&1; then
  echo "[cpp] hrm_core build + tests"
  cmake -S hrm_core -B hrm_core/build -DCMAKE_BUILD_TYPE=Release
  cmake --build hrm_core/build -j
  ctest --test-dir hrm_core/build
else
  echo "[skip] cmake not found; skipping hrm_core build/tests"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERR: python3/python not found; cannot run python tests" >&2
  exit 1
fi
echo "[python] regression tests"
"$PYTHON_BIN" -m unittest tests.test_regressions
echo "[python] silent mode tests"
"$PYTHON_BIN" -m unittest tests.test_silent_mode
echo "[python] three-mode tests"
"$PYTHON_BIN" -m unittest tests.test_modes
echo "[python] serve mode tests"
"$PYTHON_BIN" -m unittest tests.test_serve_modes
echo "[python] mode audit runtime tests"
"$PYTHON_BIN" -m unittest tests.test_mode_audit_runtime
echo "[python] router logic tests"
"$PYTHON_BIN" -m unittest tests.test_router_logic
echo "[python] router source transparency tests"
"$PYTHON_BIN" -m unittest tests.test_router_source_transparency
echo "[python] hrm_api test"
"$PYTHON_BIN" -m tests.test_hrm_api

echo "[bash] topology detection test"
bash scripts/test_topology_detection.sh

echo "[bash] native deepseek lock test"
bash scripts/test_native_lock.sh

echo "[bash] router source verify"
bash scripts/verify_router_source.sh

if command -v flash-kernel-test >/dev/null 2>&1 && "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
  echo "[cuda] flash kernel tests"
  flash-kernel-test
  flash-append-test
else
  echo "[skip] torch flash tests unavailable (missing torch or flash-kernel-test)"
fi

if [[ "${RUN_HARD_SCRIPT_TESTS:-0}" == "1" ]]; then
  echo "[bash] prod_preflight hard test"
  bash scripts/test_prod_preflight.sh
  echo "[bash] prod_live_e2e hard test"
  bash scripts/test_prod_live_e2e.sh
  echo "[bash] benchmark_deepseek hard test"
  bash scripts/test_benchmark_deepseek.sh
else
  echo "[skip] hard script tests disabled (set RUN_HARD_SCRIPT_TESTS=1 to enable)"
fi
