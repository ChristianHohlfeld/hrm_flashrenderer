#!/usr/bin/env bash
set -euo pipefail

# Build/export helper for native DeepSeek INT8 engine path.
#
# Usage:
#   scripts/build_deepseek_native.sh <model_source> [model_bin]
#
# model_source: HF repo id or local HF model directory
# model_bin: optional output .bin path (default: ./llm_models/<safe>/model_q8.bin)
#
# Optional env:
#   CUDA_ARCH_LIST (default: 75,86 for mixed 2080/3080 hosts)
#   SM (single-arch fallback, default: 75)
#   FORCE_REBUILD (0|1)
#   MODEL_QUANT (q8|q4, default: q8)

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_source> [model_bin]" >&2
  exit 2
fi

MODEL_SOURCE="$1"
MODEL_QUANT="${MODEL_QUANT:-q8}"
if [[ "$MODEL_QUANT" != "q8" && "$MODEL_QUANT" != "q4" ]]; then
  echo "ERR: MODEL_QUANT must be q8 or q4 (got '$MODEL_QUANT')" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_SH="$ROOT_DIR/scripts/deepseek_native_engine.sh"

if [[ ! -f "$RUN_SH" ]]; then
  echo "ERR: missing run script: $RUN_SH" >&2
  exit 1
fi

safe_name="${MODEL_SOURCE//\//--}"
safe_name="${safe_name//\\//--}"
safe_name="${safe_name//:/_}"

DEFAULT_BIN="$ROOT_DIR/llm_models/$safe_name/model_${MODEL_QUANT}.bin"
MODEL_BIN="${2:-$DEFAULT_BIN}"

ENGINE_BIN="${ENGINE_BIN:-$ROOT_DIR/.run/bin/deepseek_engine}"
SM="${SM:-75}"
CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-75,86}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

mkdir -p "$(dirname "$MODEL_BIN")"

echo "[build] model_source=$MODEL_SOURCE"
echo "[build] model_quant=$MODEL_QUANT"
echo "[build] model_bin=$MODEL_BIN"
echo "[build] engine_bin=$ENGINE_BIN"
echo "[build] cuda_arch_list=$CUDA_ARCH_LIST"

WORKDIR="$ROOT_DIR" \
MODEL_REPO="$MODEL_SOURCE" \
MODEL_QUANT="$MODEL_QUANT" \
MODEL_BIN="$MODEL_BIN" \
ENGINE_BIN="$ENGINE_BIN" \
SM="$SM" \
CUDA_ARCH_LIST="$CUDA_ARCH_LIST" \
FORCE_REBUILD="$FORCE_REBUILD" \
SKIP_RUN=1 \
bash "$RUN_SH"

echo "[build] done"
