#!/usr/bin/env bash
set -euo pipefail

# One-command starter with fixed hardware presets.
#
# Usage:
#   scripts/start_easy.sh <hrm_model_dir> [A|B|C|D] [q8|q4]
#
# Presets:
#   A: 22GB + 11GB + 11GB (3 GPUs), no NVLink required, no 3080 lane
#   B: 22GB + 11GB + 11GB + 10GB (4 GPUs), no NVLink required, 3080 lane enabled
#   C: 22GB + 11GB + 11GB + 10GB (4 GPUs), NVLink required on 11GB pair, 3080 lane enabled
#   D: 22GB + 22GB + 11GB + 11GB (4 GPUs), no 3080 lane
#
# Quant:
#   q8 -> defaults to DeepSeek-R1-Distill-Qwen-32B
#   q4 -> defaults to DeepSeek-R1-Distill-Llama-70B

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <hrm_model_dir> [A|B|C|D] [q8|q4]" >&2
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
  q8|q4) ;;
  *)
    echo "ERR: unsupported quant '$MODEL_QUANT' (use q8 or q4)." >&2
    exit 2
    ;;
esac

ENABLE_SOLO_3080="0"
REQUIRE_NVLINK="0"
case "$PROFILE" in
  A)
    ENABLE_SOLO_3080="0"
    REQUIRE_NVLINK="0"
    ;;
  B)
    ENABLE_SOLO_3080="1"
    REQUIRE_NVLINK="0"
    ;;
  C)
    ENABLE_SOLO_3080="1"
    REQUIRE_NVLINK="1"
    ;;
  D)
    ENABLE_SOLO_3080="0"
    REQUIRE_NVLINK="0"
    ;;
esac

echo "[easy] profile=$PROFILE quant=$MODEL_QUANT enable_solo_3080=$ENABLE_SOLO_3080 require_nvlink=$REQUIRE_NVLINK"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HW_BASE_PROFILE="$PROFILE" \
  MODEL_QUANT="$MODEL_QUANT" \
  ENABLE_SOLO_3080="$ENABLE_SOLO_3080" \
  REQUIRE_NVLINK="$REQUIRE_NVLINK" \
  bash "$SCRIPT_DIR/start_native_stack.sh" "$HRM_MODEL" auto
