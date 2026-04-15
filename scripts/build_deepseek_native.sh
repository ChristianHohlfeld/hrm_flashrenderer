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
#   MODEL_QUANT (q8 only, default: q8)
#   ENGINE_BUILD_STAMP (default: <engine_bin>.build.env)

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_source> [model_bin]" >&2
  exit 2
fi

MODEL_SOURCE="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_SH="$ROOT_DIR/scripts/deepseek_native_engine.sh"
HW_LIB="$ROOT_DIR/scripts/hw_profile_lib.sh"

if [[ ! -f "$HW_LIB" ]]; then
  echo "ERR: missing hardware profile helper: $HW_LIB" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$HW_LIB"
load_hw_selection_or_die
derive_hw_runtime_flags
print_hw_selection_summary

MODEL_QUANT="${MODEL_QUANT:-$HW_DERIVED_MODEL_QUANT}"
if [[ "$MODEL_QUANT" != "q8" ]]; then
  echo "ERR: MODEL_QUANT must be q8 (native q4 is not enabled in production mainline; got '$MODEL_QUANT')" >&2
  exit 2
fi

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
CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-$HW_DERIVED_CUDA_ARCH_LIST}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
ENGINE_BUILD_STAMP="${ENGINE_BUILD_STAMP:-${ENGINE_BIN}.build.env}"

mkdir -p "$(dirname "$MODEL_BIN")"
mkdir -p "$(dirname "$ENGINE_BUILD_STAMP")"

RUN_SH_HASH="unknown"
if command -v sha256sum >/dev/null 2>&1; then
  RUN_SH_HASH="$(sha256sum "$RUN_SH" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  RUN_SH_HASH="$(shasum -a 256 "$RUN_SH" | awk '{print $1}')"
fi

BUILD_SIGNATURE="$(cat <<EOF
HW_POOL_VERSION=${HW_POOL_VERSION}
HW_CPU_PLATFORM=${HW_CPU_PLATFORM}
HW_GPU_2080TI_11GB_COUNT=${HW_GPU_2080TI_11GB_COUNT}
HW_GPU_2080TI_22GB_COUNT=${HW_GPU_2080TI_22GB_COUNT}
HW_GPU_3080TI_10GB_COUNT=${HW_GPU_3080TI_10GB_COUNT}
HW_GPU_TOTAL=${HW_GPU_TOTAL}
HW_DERIVED_TOPOLOGY_MODE=${HW_DERIVED_TOPOLOGY_MODE}
HW_DERIVED_ENABLE_SOLO_3080=${HW_DERIVED_ENABLE_SOLO_3080}
HW_DERIVED_HW_BASE_PROFILE=${HW_DERIVED_HW_BASE_PROFILE}
MODEL_QUANT=${MODEL_QUANT}
CUDA_ARCH_LIST=${CUDA_ARCH_LIST}
ENGINE_SOURCE_SHA256=${RUN_SH_HASH}
EOF
)"

if [[ "$FORCE_REBUILD" != "1" && -x "$ENGINE_BIN" ]]; then
  previous_signature=""
  if [[ -f "$ENGINE_BUILD_STAMP" ]]; then
    previous_signature="$(cat "$ENGINE_BUILD_STAMP")"
  fi
  if [[ "$previous_signature" != "$BUILD_SIGNATURE" ]]; then
    echo "[build] engine signature changed -> forcing rebuild"
    FORCE_REBUILD=1
  fi
fi

echo "[build] model_source=$MODEL_SOURCE"
echo "[build] model_quant=$MODEL_QUANT"
echo "[build] model_bin=$MODEL_BIN"
echo "[build] engine_bin=$ENGINE_BIN"
echo "[build] cuda_arch_list=$CUDA_ARCH_LIST"
echo "[build] engine_build_stamp=$ENGINE_BUILD_STAMP"

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

cat > "$ENGINE_BUILD_STAMP" <<EOF
$BUILD_SIGNATURE
EOF

echo "[build] done"
