#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$ROOT_DIR/.run/test_topology_detection.$$"
FAKE_BIN="$TMP_DIR/bin"
MODEL_DIR="$TMP_DIR/model"
LOG_DIR="$TMP_DIR/logs"

mkdir -p "$FAKE_BIN" "$MODEL_DIR" "$LOG_DIR"
touch "$MODEL_DIR/router_index.bin" "$MODEL_DIR/index.sqlite"

cleanup_all() {
  cleanup_pids || true
  rm -rf "$TMP_DIR" || true
}
trap cleanup_all EXIT

cat > "$FAKE_BIN/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
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
  if [[ "${FAKE_NO_NVLINK:-0}" == "1" ]]; then
    cat <<'OUT'
        GPU0    GPU1    GPU2    GPU3    CPU Affinity
GPU0     X      PHB     PHB     PHB     0-31
GPU1    PHB      X      PHB     PHB     0-31
GPU2    PHB     PHB      X      PHB     0-31
GPU3    PHB     PHB     PHB      X      0-31
OUT
    exit 0
  fi
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

cat > "$FAKE_BIN/hrm-flash" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
# Keep a short-lived process alive so nohup/startup paths stay deterministic in tests.
sleep 15
EOF
chmod +x "$FAKE_BIN/hrm-flash"

cleanup_pids() {
  if [[ -d "$LOG_DIR" ]]; then
    for f in "$LOG_DIR"/*.pid; do
      [[ -f "$f" ]] || continue
      pid="$(cat "$f" 2>/dev/null || true)"
      if [[ -n "${pid:-}" ]]; then
        kill "$pid" >/dev/null 2>&1 || true
      fi
    done
  fi
}

run_and_capture() {
  local extra_env="$1"
  set +e
  # shellcheck disable=SC2086
  out=$(env PATH="$FAKE_BIN:$PATH" LOG_DIR="$LOG_DIR" STARTUP_WAIT_TIMEOUT_S=1 STARTUP_POLL_INTERVAL_S=1 $extra_env \
    bash "$ROOT_DIR/scripts/start_native_topology.sh" "$MODEL_DIR" auto 2>&1)
  code=$?
  set -e
  printf '%s' "$out"
  return "$code"
}

output_ok="$(run_and_capture "STRICT_GPU_TOPOLOGY=1" || true)"
cleanup_pids
echo "$output_ok" | grep -q "\[gpu-map\] detected pair=1,2 link=NVLINK nvlink_detected=1 solo_22gb=0 solo_3080=3 strict=1"
echo "$output_ok" | grep -q "\[gpu-map\] final mapping nvlink_pair=1,2 solo_22gb=0 solo_3080=3"

output_mismatch="$(run_and_capture "STRICT_GPU_TOPOLOGY=1 GPU_NVLINK_PAIR=0,1" || true)"
cleanup_pids
echo "$output_mismatch" | grep -q "mismatches detected NVLink pair=1,2"

output_no_nvlink="$(run_and_capture "STRICT_GPU_TOPOLOGY=1 FAKE_NO_NVLINK=1" || true)"
cleanup_pids
echo "$output_no_nvlink" | grep -q "\[warn\] no NVLink pair detected; using PCIe pair fallback: 1,2"
echo "$output_no_nvlink" | grep -q "\[gpu-map\] detected pair=1,2 link=PCIE nvlink_detected=0 solo_22gb=0 solo_3080=3 strict=1"

output_require_nvlink="$(run_and_capture "STRICT_GPU_TOPOLOGY=1 FAKE_NO_NVLINK=1 REQUIRE_NVLINK=1" || true)"
cleanup_pids
echo "$output_require_nvlink" | grep -q "ERR: no NVLink pair detected, but REQUIRE_NVLINK=1."

echo "[ok] topology detection + strict validation + PCIe fallback"
