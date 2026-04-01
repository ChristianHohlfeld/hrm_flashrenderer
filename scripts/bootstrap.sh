#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${PROFILE:-deepseek_int8}"  # deepseek_int8 | torch_tp

echo "[1/4] Build HRM core (Release)"
cmake -S "$ROOT/hrm_core" -B "$ROOT/hrm_core/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/hrm_core/build" -j

echo "[2/4] Run HRM unit tests"
ctest --test-dir "$ROOT/hrm_core/build"

echo "[3/4] Install Python dependencies (profile=$PROFILE)"
python -m pip install -r "$ROOT/requirements.prod.txt"
python -m pip install -r "$ROOT/requirements.server.txt"
if [[ "$PROFILE" == "torch_tp" ]]; then
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.6}"
  python -m pip install -r "$ROOT/requirements.torch.txt"
fi
python -m pip install -e "$ROOT"

if [[ "$PROFILE" == "torch_tp" ]]; then
  echo "[4/4] Flash kernel tests"
  flash-kernel-test
  flash-append-test
else
  echo "[4/4] Skip torch kernel tests for profile=$PROFILE"
fi

echo "OK"
