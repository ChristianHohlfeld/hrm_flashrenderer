#!/usr/bin/env bash
set -euo pipefail

# Production preflight for native DeepSeek stack rollout.
#
# Usage:
#   scripts/prod_preflight.sh <hrm_model_dir> [expected_gpu_count]
#
# Example:
#   scripts/prod_preflight.sh ./model_index 4

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <hrm_model_dir> [expected_gpu_count]" >&2
  exit 2
fi

HRM_MODEL="$1"
EXPECTED_GPUS="${2:-4}"

fail() {
  echo "ERR: $*" >&2
  exit 1
}

check_cmd() {
  local c="$1"
  command -v "$c" >/dev/null 2>&1 || fail "required command not found: $c"
}

echo "[preflight] checking required commands..."
check_cmd python3
check_cmd bash
check_cmd curl
check_cmd nvidia-smi
check_cmd nvcc

echo "[preflight] checking Python version..."
python3 - <<'PY'
import sys
v = sys.version_info
ok = (v.major == 3 and 10 <= v.minor <= 12)
print(f"python={v.major}.{v.minor}.{v.micro}")
if not ok:
    raise SystemExit("unsupported Python version for pinned deps; use Python 3.10-3.12")
PY

echo "[preflight] checking HRM query backend..."
if [[ -x "hrm_core/build/hrm" ]]; then
  echo "[preflight] using repo HRM binary: hrm_core/build/hrm"
elif command -v hrm >/dev/null 2>&1; then
  echo "[preflight] using PATH HRM binary: $(command -v hrm)"
else
  if python3 - <<'PY'
from pathlib import Path
ok = False
for p in (
    Path("hrm_core/build/libhrm_api.so"),
    Path("/usr/local/lib/libhrm_api.so"),
    Path("/usr/lib/libhrm_api.so"),
):
    if p.is_file():
        ok = True
        break
raise SystemExit(0 if ok else 1)
PY
  then
    echo "[preflight] using libhrm_api.so fallback"
  else
    fail "no HRM backend found (need hrm binary or libhrm_api.so)"
  fi
fi

echo "[preflight] checking HRM model index..."
[[ -f "$HRM_MODEL/router_index.bin" ]] || fail "missing $HRM_MODEL/router_index.bin"
[[ -f "$HRM_MODEL/index.sqlite" ]] || fail "missing $HRM_MODEL/index.sqlite"

echo "[preflight] checking repository scripts..."
[[ -f "scripts/build_deepseek_native.sh" ]] || fail "missing scripts/build_deepseek_native.sh"
[[ -f "scripts/deepseek_native_engine.sh" ]] || fail "missing scripts/deepseek_native_engine.sh"
[[ -f "scripts/start_native_stack.sh" ]] || fail "missing scripts/start_native_stack.sh"
[[ -f "scripts/stop_native_stack.sh" ]] || fail "missing scripts/stop_native_stack.sh"

echo "[preflight] checking visible GPUs..."
gpu_count="$(nvidia-smi -L | wc -l | tr -d ' ')"
if [[ -z "$gpu_count" || "$gpu_count" -lt 1 ]]; then
  fail "no GPUs reported by nvidia-smi"
fi
if [[ "$gpu_count" -lt "$EXPECTED_GPUS" ]]; then
  fail "expected at least $EXPECTED_GPUS GPUs, found $gpu_count"
fi
echo "[preflight] gpu_count=$gpu_count"

echo "[preflight] checking python deps (lean + server)..."
python3 - <<'PY'
import importlib
mods = ["numpy", "huggingface_hub", "transformers", "safetensors", "fastapi", "uvicorn", "pydantic"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit("missing python modules: " + ", ".join(missing))
print("python deps OK")
PY

echo "[preflight] checking CLI entrypoints..."
python3 -m hrm_flash.cli --help >/dev/null
python3 -m hrm_flash.cli serve --help >/dev/null
python3 -m hrm_flash.cli router --help >/dev/null
python3 -m hrm_flash.cli generate --help >/dev/null

echo "[preflight] checking script syntax..."
bash -n scripts/build_deepseek_native.sh
bash -n scripts/start_native_topology.sh
bash -n scripts/start_native_stack.sh
bash -n scripts/stop_native_topology.sh
bash -n scripts/stop_native_stack.sh
bash -n scripts/smoke_router.sh
bash -n scripts/deepseek_native_engine.sh
bash -n scripts/test_topology_detection.sh
bash -n scripts/prod_live_e2e.sh

echo "[preflight] running topology detection self-test..."
bash scripts/test_topology_detection.sh

echo "[preflight] checking GPU topology (informational)..."
if nvidia-smi topo -m >/tmp/hrm_topo.txt 2>/dev/null; then
  grep -E "GPU[0-9]|NV" /tmp/hrm_topo.txt || true
  rm -f /tmp/hrm_topo.txt
else
  echo "[preflight] nvidia-smi topo -m not available; skipping."
fi

echo
echo "Preflight OK. Ready for native production rollout."
