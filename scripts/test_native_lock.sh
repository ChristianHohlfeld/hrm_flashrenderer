#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d "${ROOT_DIR}/.run/test_native_lock.XXXXXX")"
MODEL_DIR="$TMP_DIR/model"
HW_FILE="$TMP_DIR/hw_selection.env"
mkdir -p "$MODEL_DIR"
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

expect_fail_contains() {
  local needle="$1"
  shift
  set +e
  local out
  out="$("$@" 2>&1)"
  local code=$?
  set -e
  if [[ "$code" -eq 0 ]]; then
    echo "ERR: command unexpectedly succeeded: $*" >&2
    exit 1
  fi
  printf '%s' "$out" | grep -q "$needle" || {
    echo "ERR: expected failure output to contain '$needle'" >&2
    echo "$out" >&2
    exit 1
  }
}

echo "[test] bootstrap rejects torch profile"
expect_fail_contains "not supported in production mainline" env PROFILE=torch_tp bash "$ROOT_DIR/scripts/bootstrap.sh"

echo "[test] start_native_stack rejects non-deepseek backend"
expect_fail_contains "not supported in production mainline" env HW_SELECTION_FILE="$HW_FILE" BACKEND=torch_tp PREPARE_MODELS=0 bash "$ROOT_DIR/scripts/start_native_stack.sh" "$MODEL_DIR" auto

echo "[test] start_native_topology rejects non-deepseek backend"
expect_fail_contains "not supported in production mainline" env HW_SELECTION_FILE="$HW_FILE" BACKEND=torch_tp bash "$ROOT_DIR/scripts/start_native_topology.sh" "$MODEL_DIR" auto

echo "[test] start scripts reject unsupported topology mode"
expect_fail_contains "conflicts with selected hardware pool" env HW_SELECTION_FILE="$HW_FILE" PREPARE_MODELS=0 TOPOLOGY_MODE=invalid bash "$ROOT_DIR/scripts/start_native_stack.sh" "$MODEL_DIR" auto
expect_fail_contains "conflicts with selected hardware pool" env HW_SELECTION_FILE="$HW_FILE" TOPOLOGY_MODE=invalid bash "$ROOT_DIR/scripts/start_native_topology.sh" "$MODEL_DIR" auto

echo "[test] easy launcher rejects invalid preset"
expect_fail_contains "unsupported profile" bash "$ROOT_DIR/scripts/start_easy.sh" "$MODEL_DIR" Z q8

echo "[test] hw selection rejects q4 in production mainline"
expect_fail_contains "must be q8" bash "$ROOT_DIR/scripts/hw_select.sh" --gpu-2080ti-11gb 2 --gpu-2080ti-22gb 1 --gpu-3080ti-10gb 1 --model-quant q4

echo "[test] start scripts reject unsupported DeepSeek model sizes"
expect_fail_contains "unsupported deepseek model" env HW_SELECTION_FILE="$HW_FILE" PREPARE_MODELS=0 LLM_MODEL_TRIPLE_MAX=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B bash "$ROOT_DIR/scripts/start_native_stack.sh" "$MODEL_DIR" auto

echo "[test] serve helper rejects non-deepseek backend"
expect_fail_contains "not supported in production mainline" env BACKEND=torch_tp bash "$ROOT_DIR/scripts/serve.sh" "$MODEL_DIR" "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" 18080

echo "[test] generate helper rejects non-deepseek backend"
expect_fail_contains "not supported in production mainline" env BACKEND=torch_tp bash "$ROOT_DIR/scripts/hrm_flash_generate.sh" "$MODEL_DIR" "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" "Hallo"

echo "[test] cli serve parser rejects torch_tp backend"
expect_fail_contains "invalid choice" python3 -m hrm_flash.cli serve --hrm_model "$MODEL_DIR" --llm_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" --world 1 --backend torch_tp

echo "[test] cli generate parser rejects torch_tp backend"
expect_fail_contains "invalid choice" python3 -m hrm_flash.cli generate --hrm_model "$MODEL_DIR" --llm_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" --prompt "Hallo" --backend torch_tp

echo "[test] cli serve parser rejects q4 model quant"
expect_fail_contains "invalid choice" python3 -m hrm_flash.cli serve --hrm_model "$MODEL_DIR" --llm_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" --world 1 --model_quant q4

echo "[ok] native deepseek-only lock guards"
