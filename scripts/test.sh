#!/usr/bin/env bash
set -euo pipefail
cmake -S hrm_core -B hrm_core/build -DCMAKE_BUILD_TYPE=Release
cmake --build hrm_core/build -j
ctest --test-dir hrm_core/build

echo "[python] hrm_api test"
python -m tests.test_hrm_api

if command -v flash-kernel-test >/dev/null 2>&1; then
  echo "[cuda] flash kernel tests"
  flash-kernel-test
  flash-append-test
fi
