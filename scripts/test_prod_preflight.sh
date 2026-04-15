#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d "${ROOT_DIR}/.run/test_prod_preflight.XXXXXX")"
FAKE_BIN="$TMP_DIR/bin"
MODEL_DIR="$TMP_DIR/model"
HW_FILE="$TMP_DIR/hw_selection.env"

mkdir -p "$FAKE_BIN" "$MODEL_DIR"
touch "$MODEL_DIR/router_index.bin" "$MODEL_DIR/index.sqlite"
cat > "$HW_FILE" <<'EOF'
HW_POOL_VERSION=1
HW_CPU_PLATFORM="xeon_e5-2680_v4_256gb_ddr4"
HW_GPU_2080TI_11GB_COUNT=2
HW_GPU_2080TI_22GB_COUNT=1
HW_GPU_3080TI_10GB_COUNT=1
HW_REQUIRE_NVLINK_11GB_PAIR=1
HW_MODEL_QUANT="q8"
EOF

cleanup() {
  rm -rf "$TMP_DIR" || true
}
trap cleanup EXIT

cat > "$FAKE_BIN/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-L" ]]; then
  cat <<'OUT'
GPU 0: Mock 22GB
GPU 1: Mock 11GB
GPU 2: Mock 11GB
GPU 3: Mock 10GB
OUT
  exit 0
fi
if [[ "${1:-}" == "--query-gpu=index,memory.total" ]]; then
  cat <<'OUT'
0, 22528
1, 11264
2, 11264
3, 10240
OUT
  exit 0
fi
if [[ "${1:-}" == "topo" && "${2:-}" == "-m" ]]; then
  cat <<'OUT'
        GPU0    GPU1    GPU2    GPU3    CPU Affinity
GPU0     X      PHB     PHB     PHB     0-31
GPU1    PHB      X      NV2     PHB     0-31
GPU2    PHB     NV2      X      PHB     0-31
GPU3    PHB     PHB     PHB      X      0-31
OUT
  exit 0
fi
echo "unsupported fake nvidia-smi args: $*" >&2
exit 1
EOF
chmod +x "$FAKE_BIN/nvidia-smi"

cat > "$FAKE_BIN/nvcc" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "Cuda compilation tools, release 12.4, V12.4.99"
EOF
chmod +x "$FAKE_BIN/nvcc"

cat > "$FAKE_BIN/hrm" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "mock hrm"
EOF
chmod +x "$FAKE_BIN/hrm"

run_preflight() {
  PATH="$FAKE_BIN:$PATH" \
  HW_SELECTION_FILE="$HW_FILE" \
  SKIP_PY_DEPS_CHECK=1 \
  bash "$ROOT_DIR/scripts/prod_preflight.sh" "$MODEL_DIR" "$1"
}

echo "[test] preflight happy path"
run_preflight 4 >/dev/null

echo "[test] preflight fails on impossible GPU expectation"
set +e
out_fail_gpu="$(run_preflight 8 2>&1)"
code_fail_gpu=$?
set -e
if [[ "$code_fail_gpu" -eq 0 ]]; then
  echo "ERR: preflight unexpectedly succeeded for EXPECTED_GPUS=8" >&2
  exit 1
fi
printf '%s' "$out_fail_gpu" | grep -q "expected at least 8 GPUs"

echo "[test] preflight fails on missing index file"
rm -f "$MODEL_DIR/index.sqlite"
set +e
out_fail_model="$(run_preflight 4 2>&1)"
code_fail_model=$?
set -e
if [[ "$code_fail_model" -eq 0 ]]; then
  echo "ERR: preflight unexpectedly succeeded without index.sqlite" >&2
  exit 1
fi
printf '%s' "$out_fail_model" | grep -q "missing .*index.sqlite"

echo "[ok] prod_preflight hard checks"
