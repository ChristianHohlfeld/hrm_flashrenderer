#!/usr/bin/env bash
set -euo pipefail

# One-command starter with fixed hardware presets.
#
# Usage:
#   scripts/start_easy.sh <hrm_model_dir> [A|B|C|D] [q8]
#
# Presets:
#   A: 22GB + 11GB + 11GB (3 GPUs), no NVLink required, no 3080 lane
#   B: 22GB + 11GB + 11GB + 10GB (4 GPUs), no NVLink required, 3080 lane enabled
#   C: 22GB + 11GB + 11GB + 10GB (4 GPUs), NVLink required on 11GB pair, 3080 lane enabled
#   D: 22GB + 22GB + 11GB + 11GB (4 GPUs), no 3080 lane
#
# Quant:
#   q8 only (native q4 is not enabled in production mainline)

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <hrm_model_dir> [A|B|C|D] [q8]" >&2
  exit 2
fi

HRM_MODEL="$1"
PROFILE_RAW="${2:-A}"
MODEL_QUANT="${3:-q8}"

PROFILE="$(printf '%s' "$PROFILE_RAW" | tr '[:lower:]' '[:upper:]')"
case "$PROFILE" in
  A|B|C|D) ;;
  *)
    echo "ERR: unsupported profile '$PROFILE_RAW' (use A, B, C, or D)." >&2
    exit 2
    ;;
esac

case "$MODEL_QUANT" in
  q8) ;;
  *)
    echo "ERR: unsupported quant '$MODEL_QUANT' (use q8; native q4 is not enabled in production mainline)." >&2
    exit 2
    ;;
esac

REQUIRE_NVLINK="0"
GPU11_COUNT="2"
GPU22_COUNT="1"
GPU3080_COUNT="0"
case "$PROFILE" in
  A)
    GPU11_COUNT="2"
    GPU22_COUNT="1"
    GPU3080_COUNT="0"
    REQUIRE_NVLINK="0"
    ;;
  B)
    GPU11_COUNT="2"
    GPU22_COUNT="1"
    GPU3080_COUNT="1"
    REQUIRE_NVLINK="0"
    ;;
  C)
    GPU11_COUNT="2"
    GPU22_COUNT="1"
    GPU3080_COUNT="1"
    REQUIRE_NVLINK="1"
    ;;
  D)
    GPU11_COUNT="2"
    GPU22_COUNT="2"
    GPU3080_COUNT="0"
    REQUIRE_NVLINK="0"
    ;;
esac

echo "[easy] profile=$PROFILE quant=$MODEL_QUANT pool=(11gb:$GPU11_COUNT,22gb:$GPU22_COUNT,3080_10gb:$GPU3080_COUNT) require_nvlink=$REQUIRE_NVLINK"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HW_SELECTION_FILE="${HW_SELECTION_FILE:-}" \
  bash "$SCRIPT_DIR/hw_select.sh" \
    --gpu-2080ti-11gb "$GPU11_COUNT" \
    --gpu-2080ti-22gb "$GPU22_COUNT" \
    --gpu-3080ti-10gb "$GPU3080_COUNT" \
    --require-nvlink-11gb-pair "$REQUIRE_NVLINK" \
    --model-quant "$MODEL_QUANT"

REQUIRE_NVLINK="$REQUIRE_NVLINK" \
  bash "$SCRIPT_DIR/start_native_stack.sh" "$HRM_MODEL" auto
