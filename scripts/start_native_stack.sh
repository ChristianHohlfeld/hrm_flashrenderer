#!/usr/bin/env bash
set -euo pipefail

# Starts full native heterogeneous stack:
# 1) native services according to TOPOLOGY_MODE
# 2) one hrm-flash router endpoint with auto-dispatch + failover
#
# Usage:
#   scripts/start_native_stack.sh <hrm_model_dir> [default_llm_model|auto]
#
# Important env vars (optional):
#   BACKEND (default: deepseek_int8)
#   TOPOLOGY_MODE (default: max_model_fast; supported: max_model_fast, hetero_3lane)
#   MODEL_PROFILE (default: max_vram_hetero for auto mode)
#   HW_BASE_PROFILE (default: auto; auto|A|B|C|D hardware presets)
#   LLM_MODEL_SOLO_22GB / LLM_MODEL_NVLINK_PAIR / LLM_MODEL_SOLO_3080
#   LLM_MODEL_TRIPLE_MAX
#   ENABLE_SOLO_3080 (default: derived from HW_BASE_PROFILE; set 1 to force dedicated 3080 lane)
#   STRICT_GPU_TOPOLOGY (default: 1, enforced by start_native_topology.sh)
#   PORT_SOLO_22GB, PORT_NVLINK_PAIR, PORT_SOLO_3080
#   ROUTER_HOST, ROUTER_PORT
#   ROUTER_MAX_CONCURRENT
#   ROUTER_SHORT_PROMPT_TOKENS, ROUTER_MEDIUM_PROMPT_TOKENS
#   ROUTER_SHORT_MAX_NEW_TOKENS, ROUTER_MEDIUM_MAX_NEW_TOKENS, ROUTER_LONG_MAX_NEW_TOKENS
#   ROUTER_REQUEST_TIMEOUT_S, ROUTER_HEALTH_TIMEOUT_S
#   ROUTER_TOKENIZER_MODEL
#   ROUTER_DEFAULT_MODE (default: mixed; retrieval|mixed|deepseek_only)
#   ROUTER_LOCAL_FILES_ONLY (default: LOCAL_FILES_ONLY or 1)
#   ROUTER_DISABLE_TOKENIZER (1 disables tokenizer-based routing)
#   PREPARE_MODELS (default: 1 for deepseek_int8 backend)
#   STARTUP_WAIT_TIMEOUT_S, STARTUP_POLL_INTERVAL_S (forwarded to topology and router wait)
#   LOG_DIR
#   PYTHON_BIN (default: python3, fallback: python)
#   HRM_FLASH_BIN (optional: explicit hrm-flash executable override)

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <hrm_model_dir> [default_llm_model|auto]" >&2
  exit 2
fi

HRM_MODEL="$1"
DEFAULT_LLM_MODEL="${2:-auto}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

LOG_DIR="${LOG_DIR:-$ROOT_DIR/.run/services}"
mkdir -p "$LOG_DIR"
BACKEND="${BACKEND:-deepseek_int8}"
TOPOLOGY_MODE="${TOPOLOGY_MODE:-max_model_fast}"
PREPARE_MODELS="${PREPARE_MODELS:-1}"
MODEL_PROFILE="${MODEL_PROFILE:-max_vram_hetero}"
HW_BASE_PROFILE="${HW_BASE_PROFILE:-auto}"
MODEL_QUANT="${MODEL_QUANT:-q8}"
STRICT_GPU_TOPOLOGY="${STRICT_GPU_TOPOLOGY:-1}"
STARTUP_WAIT_TIMEOUT_S="${STARTUP_WAIT_TIMEOUT_S:-240}"
STARTUP_POLL_INTERVAL_S="${STARTUP_POLL_INTERVAL_S:-2}"
STARTUP_WAIT_TIMEOUT_S="${STARTUP_WAIT_TIMEOUT_S%.*}"
STARTUP_POLL_INTERVAL_S="${STARTUP_POLL_INTERVAL_S%.*}"
if [[ -z "${STARTUP_WAIT_TIMEOUT_S:-}" || "$STARTUP_WAIT_TIMEOUT_S" -lt 1 ]]; then
  STARTUP_WAIT_TIMEOUT_S=240
fi
if [[ -z "${STARTUP_POLL_INTERVAL_S:-}" || "$STARTUP_POLL_INTERVAL_S" -lt 1 ]]; then
  STARTUP_POLL_INTERVAL_S=2
fi

if [[ "$BACKEND" != "deepseek_int8" ]]; then
  echo "ERR: BACKEND=$BACKEND is not supported in production mainline. Use deepseek_int8." >&2
  exit 2
fi
if [[ "$MODEL_QUANT" != "q8" && "$MODEL_QUANT" != "q4" ]]; then
  echo "ERR: unsupported MODEL_QUANT=$MODEL_QUANT (supported: q8, q4)." >&2
  exit 2
fi
if [[ "$TOPOLOGY_MODE" != "max_model_fast" && "$TOPOLOGY_MODE" != "hetero_3lane" ]]; then
  echo "ERR: unsupported TOPOLOGY_MODE=$TOPOLOGY_MODE (supported: max_model_fast, hetero_3lane)." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "ERR: python3/python not found. Install Python 3.10-3.12." >&2
    exit 1
  fi
fi
if [[ -n "${HRM_FLASH_BIN:-}" ]]; then
  HRM_FLASH_CMD=("$HRM_FLASH_BIN")
else
  # Force local repo code path so mode/prompt logic cannot silently drift to a stale global install.
  export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
  HRM_FLASH_CMD=("$PYTHON_BIN" -m hrm_flash.cli)
fi

if [[ -z "${HRM_FLASH_BIN:-}" ]]; then
  ROUTER_SOURCE_PATH="$(
    "$PYTHON_BIN" - <<'PY'
import inspect
import pathlib
import py_compile

import hrm_flash.prompt_builder as prompt_builder
import hrm_flash.router as router
import hrm_flash.serve as serve

modules = [router, serve, prompt_builder]
for mod in modules:
    src = pathlib.Path(inspect.getsourcefile(mod) or inspect.getfile(mod)).resolve()
    py_compile.compile(str(src), doraise=True)

router_src = pathlib.Path(inspect.getsourcefile(router) or inspect.getfile(router)).resolve()
print(str(router_src))
PY
  )"
  echo "[stack] router source=$ROUTER_SOURCE_PATH (compile-checked)"
else
  echo "[stack] router source=external binary (\$HRM_FLASH_BIN) - source visibility depends on that binary"
fi

PORT_SOLO_22GB="${PORT_SOLO_22GB:-8081}"
PORT_NVLINK_PAIR="${PORT_NVLINK_PAIR:-8082}"
PORT_SOLO_3080="${PORT_SOLO_3080:-8083}"

RECO_MODEL_SOLO_22GB="${RECO_MODEL_SOLO_22GB:-deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}"
RECO_MODEL_NVLINK_PAIR="${RECO_MODEL_NVLINK_PAIR:-deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}"
RECO_MODEL_SOLO_3080="${RECO_MODEL_SOLO_3080:-deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}"
RECO_MODEL_TRIPLE_MAX_Q8="${RECO_MODEL_TRIPLE_MAX_Q8:-deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}"
RECO_MODEL_TRIPLE_MAX_Q4="${RECO_MODEL_TRIPLE_MAX_Q4:-deepseek-ai/DeepSeek-R1-Distill-Llama-70B}"
if [[ -z "${RECO_MODEL_TRIPLE_MAX:-}" ]]; then
  if [[ "$MODEL_QUANT" == "q4" ]]; then
    RECO_MODEL_TRIPLE_MAX="$RECO_MODEL_TRIPLE_MAX_Q4"
  else
    RECO_MODEL_TRIPLE_MAX="$RECO_MODEL_TRIPLE_MAX_Q8"
  fi
fi

use_profile=0
case "$DEFAULT_LLM_MODEL" in
  ""|auto|max|max_vram|max_vram_hetero)
    use_profile=1
    ;;
esac

if [[ "$use_profile" == "1" ]]; then
  if [[ "$BACKEND" != "deepseek_int8" ]]; then
    echo "ERR: auto model profile requires BACKEND=deepseek_int8. Pass explicit model for backend=$BACKEND." >&2
    exit 2
  fi
  if [[ "$MODEL_PROFILE" != "max_vram_hetero" ]]; then
    echo "ERR: unsupported MODEL_PROFILE=$MODEL_PROFILE (supported: max_vram_hetero)." >&2
    exit 2
  fi
  if [[ "$TOPOLOGY_MODE" == "max_model_fast" ]]; then
    LLM_MODEL_TRIPLE_MAX="${LLM_MODEL_TRIPLE_MAX:-$RECO_MODEL_TRIPLE_MAX}"
    LLM_MODEL_SOLO_3080="${LLM_MODEL_SOLO_3080:-$RECO_MODEL_SOLO_3080}"
    # Router keeps three logical lanes; quality and balanced can point to the same max-model service.
    PORT_NVLINK_PAIR="$PORT_SOLO_22GB"
    LLM_MODEL_SOLO_22GB="$LLM_MODEL_TRIPLE_MAX"
    LLM_MODEL_NVLINK_PAIR="$LLM_MODEL_TRIPLE_MAX"
    DEFAULT_LLM_MODEL="$LLM_MODEL_TRIPLE_MAX"
  else
    LLM_MODEL_SOLO_22GB="${LLM_MODEL_SOLO_22GB:-$RECO_MODEL_SOLO_22GB}"
    LLM_MODEL_NVLINK_PAIR="${LLM_MODEL_NVLINK_PAIR:-$RECO_MODEL_NVLINK_PAIR}"
    LLM_MODEL_SOLO_3080="${LLM_MODEL_SOLO_3080:-$RECO_MODEL_SOLO_3080}"
    DEFAULT_LLM_MODEL="$LLM_MODEL_SOLO_22GB"
  fi
else
  if [[ "$TOPOLOGY_MODE" == "max_model_fast" ]]; then
    LLM_MODEL_TRIPLE_MAX="${LLM_MODEL_TRIPLE_MAX:-$DEFAULT_LLM_MODEL}"
    LLM_MODEL_SOLO_3080="${LLM_MODEL_SOLO_3080:-$RECO_MODEL_SOLO_3080}"
    PORT_NVLINK_PAIR="$PORT_SOLO_22GB"
    LLM_MODEL_SOLO_22GB="$LLM_MODEL_TRIPLE_MAX"
    LLM_MODEL_NVLINK_PAIR="$LLM_MODEL_TRIPLE_MAX"
  else
    LLM_MODEL_SOLO_22GB="${LLM_MODEL_SOLO_22GB:-$DEFAULT_LLM_MODEL}"
    LLM_MODEL_NVLINK_PAIR="${LLM_MODEL_NVLINK_PAIR:-$DEFAULT_LLM_MODEL}"
    LLM_MODEL_SOLO_3080="${LLM_MODEL_SOLO_3080:-$DEFAULT_LLM_MODEL}"
  fi
fi

case "${HW_BASE_PROFILE,,}" in
  auto|a|option_a|b|option_b|c|option_c|d|option_d) ;;
  *)
    echo "ERR: unsupported HW_BASE_PROFILE=$HW_BASE_PROFILE (supported: auto, A, B, C, D)." >&2
    exit 2
    ;;
esac

if [[ -z "${ENABLE_SOLO_3080+x}" ]]; then
  case "${HW_BASE_PROFILE,,}" in
    b|option_b|c|option_c)
      ENABLE_SOLO_3080="1"
      ;;
    *)
      ENABLE_SOLO_3080="0"
      ;;
  esac
fi
case "$ENABLE_SOLO_3080" in
  0|1) ;;
  *)
    echo "ERR: ENABLE_SOLO_3080 must be 0 or 1 (got: $ENABLE_SOLO_3080)." >&2
    exit 2
    ;;
esac

is_supported_deepseek_model() {
  local model="$1"
  case "$model" in
    deepseek-ai/DeepSeek-R1-Distill-Qwen-32B|deepseek-ai/DeepSeek-R1-Distill-Llama-70B)
      return 0
      ;;
  esac
  return 1
}

validate_supported_model() {
  local label="$1"
  local model="$2"
  if [[ -z "$model" ]]; then
    echo "ERR: $label is empty; set one of: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B or deepseek-ai/DeepSeek-R1-Distill-Llama-70B" >&2
    exit 2
  fi
  if ! is_supported_deepseek_model "$model"; then
    echo "ERR: unsupported deepseek model for $label: $model" >&2
    echo "Supported models: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B, deepseek-ai/DeepSeek-R1-Distill-Llama-70B" >&2
    exit 2
  fi
}

if [[ "$TOPOLOGY_MODE" == "max_model_fast" ]]; then
  validate_supported_model "LLM_MODEL_TRIPLE_MAX" "$LLM_MODEL_SOLO_22GB"
else
  validate_supported_model "LLM_MODEL_SOLO_22GB" "$LLM_MODEL_SOLO_22GB"
  validate_supported_model "LLM_MODEL_NVLINK_PAIR" "$LLM_MODEL_NVLINK_PAIR"
fi
if [[ "$ENABLE_SOLO_3080" == "1" ]]; then
  validate_supported_model "LLM_MODEL_SOLO_3080" "$LLM_MODEL_SOLO_3080"
fi

if [[ "$ENABLE_SOLO_3080" == "0" ]]; then
  # Keep router fast lane alive without forcing a <32B model on 10GB.
  PORT_SOLO_3080="$PORT_SOLO_22GB"
fi

echo "[stack] backend=$BACKEND topology_mode=$TOPOLOGY_MODE profile=$MODEL_PROFILE hw_base_profile=${HW_BASE_PROFILE,,} auto_profile=$use_profile quant=$MODEL_QUANT"
if [[ "$TOPOLOGY_MODE" == "max_model_fast" ]]; then
  if [[ "$ENABLE_SOLO_3080" == "1" ]]; then
    echo "[stack] max_model_fast: route default targets max-model lane; route_hint=fast uses dedicated 3080 lane"
  else
    echo "[stack] max_model_fast: route default targets max-model lane; route_hint=fast aliases to max-model lane (solo_3080 disabled)"
  fi
fi
echo "[stack] models: solo_22gb=$LLM_MODEL_SOLO_22GB nvlink_pair=$LLM_MODEL_NVLINK_PAIR solo_3080=$LLM_MODEL_SOLO_3080"

ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_PORT="${ROUTER_PORT:-8090}"
ROUTER_MAX_CONCURRENT="${ROUTER_MAX_CONCURRENT:-8}"
ROUTER_SHORT_PROMPT_TOKENS="${ROUTER_SHORT_PROMPT_TOKENS:-256}"
ROUTER_MEDIUM_PROMPT_TOKENS="${ROUTER_MEDIUM_PROMPT_TOKENS:-1200}"
ROUTER_SHORT_MAX_NEW_TOKENS="${ROUTER_SHORT_MAX_NEW_TOKENS:-192}"
ROUTER_MEDIUM_MAX_NEW_TOKENS="${ROUTER_MEDIUM_MAX_NEW_TOKENS:-384}"
ROUTER_LONG_MAX_NEW_TOKENS="${ROUTER_LONG_MAX_NEW_TOKENS:-768}"
ROUTER_REQUEST_TIMEOUT_S="${ROUTER_REQUEST_TIMEOUT_S:-180}"
ROUTER_HEALTH_TIMEOUT_S="${ROUTER_HEALTH_TIMEOUT_S:-1.5}"
ROUTER_CHARS_PER_TOKEN="${ROUTER_CHARS_PER_TOKEN:-4.0}"
ROUTER_DISABLE_TOKENIZER="${ROUTER_DISABLE_TOKENIZER:-0}"
ROUTER_TOKENIZER_MODEL="${ROUTER_TOKENIZER_MODEL:-$LLM_MODEL_SOLO_22GB}"
ROUTER_DEFAULT_MODE="${ROUTER_DEFAULT_MODE:-mixed}"
ROUTER_LOCAL_FILES_ONLY="${ROUTER_LOCAL_FILES_ONLY:-${LOCAL_FILES_ONLY:-1}}"
case "$ROUTER_DEFAULT_MODE" in
  retrieval|mixed|deepseek_only) ;;
  *)
    echo "ERR: unsupported ROUTER_DEFAULT_MODE=$ROUTER_DEFAULT_MODE (supported: retrieval, mixed, deepseek_only)." >&2
    exit 2
    ;;
esac
echo "[stack] router default_mode=$ROUTER_DEFAULT_MODE"

health_ok() {
  local url="$1"
  if ! command -v curl >/dev/null 2>&1; then
    return 1
  fi
  local out
  out="$(curl -fsS --max-time 2 "$url" 2>/dev/null || true)"
  if [[ -z "$out" ]]; then
    return 1
  fi
  if command -v python3 >/dev/null 2>&1; then
    printf '%s' "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); raise SystemExit(0 if bool(d.get('ok')) else 1)" >/dev/null 2>&1
    return $?
  fi
  [[ "$out" == *"\"ok\":true"* || "$out" == *"\"ok\": true"* ]]
}

if [[ "$BACKEND" == "deepseek_int8" && "$PREPARE_MODELS" == "1" ]]; then
  echo "[stack] preparing deepseek model bins..."
  BUILD_SCRIPT="$SCRIPT_DIR/build_deepseek_native.sh"
  if [[ ! -f "$BUILD_SCRIPT" ]]; then
    echo "ERR: missing build helper: $BUILD_SCRIPT" >&2
    exit 1
  fi
  build_one() {
    local model="$1"
    local out_bin="$2"
    if [[ -n "${out_bin:-}" ]]; then
      MODEL_QUANT="$MODEL_QUANT" bash "$BUILD_SCRIPT" "$model" "$out_bin"
    else
      MODEL_QUANT="$MODEL_QUANT" bash "$BUILD_SCRIPT" "$model"
    fi
  }
  if [[ "$TOPOLOGY_MODE" == "max_model_fast" ]]; then
    build_one "${LLM_MODEL_TRIPLE_MAX:-$LLM_MODEL_SOLO_22GB}" "${MODEL_BIN_TRIPLE_MAX:-${MODEL_BIN_SOLO_22GB:-}}"
    if [[ "$ENABLE_SOLO_3080" == "1" ]]; then
      build_one "${LLM_MODEL_SOLO_3080:-$DEFAULT_LLM_MODEL}" "${MODEL_BIN_SOLO_3080:-}"
    fi
  else
    build_one "${LLM_MODEL_SOLO_22GB:-$DEFAULT_LLM_MODEL}" "${MODEL_BIN_SOLO_22GB:-}"
    build_one "${LLM_MODEL_NVLINK_PAIR:-$DEFAULT_LLM_MODEL}" "${MODEL_BIN_NVLINK_PAIR:-}"
    if [[ "$ENABLE_SOLO_3080" == "1" ]]; then
      build_one "${LLM_MODEL_SOLO_3080:-$DEFAULT_LLM_MODEL}" "${MODEL_BIN_SOLO_3080:-}"
    fi
  fi
fi

echo "[stack] starting native topology services..."
MODEL_PROFILE="$MODEL_PROFILE" \
HW_BASE_PROFILE="$HW_BASE_PROFILE" \
TOPOLOGY_MODE="$TOPOLOGY_MODE" \
MODEL_QUANT="$MODEL_QUANT" \
ENABLE_SOLO_3080="$ENABLE_SOLO_3080" \
STRICT_GPU_TOPOLOGY="$STRICT_GPU_TOPOLOGY" \
LLM_MODEL_SOLO_22GB="$LLM_MODEL_SOLO_22GB" \
LLM_MODEL_NVLINK_PAIR="$LLM_MODEL_NVLINK_PAIR" \
LLM_MODEL_SOLO_3080="$LLM_MODEL_SOLO_3080" \
LLM_MODEL_TRIPLE_MAX="${LLM_MODEL_TRIPLE_MAX:-}" \
MODEL_BIN_SOLO_22GB="${MODEL_BIN_SOLO_22GB:-}" \
MODEL_BIN_NVLINK_PAIR="${MODEL_BIN_NVLINK_PAIR:-}" \
MODEL_BIN_SOLO_3080="${MODEL_BIN_SOLO_3080:-}" \
MODEL_BIN_TRIPLE_MAX="${MODEL_BIN_TRIPLE_MAX:-}" \
TOKENIZER_MODEL_SOLO_22GB="${TOKENIZER_MODEL_SOLO_22GB:-}" \
TOKENIZER_MODEL_NVLINK_PAIR="${TOKENIZER_MODEL_NVLINK_PAIR:-}" \
TOKENIZER_MODEL_SOLO_3080="${TOKENIZER_MODEL_SOLO_3080:-}" \
TOKENIZER_MODEL_TRIPLE_MAX="${TOKENIZER_MODEL_TRIPLE_MAX:-}" \
STARTUP_WAIT_TIMEOUT_S="$STARTUP_WAIT_TIMEOUT_S" \
STARTUP_POLL_INTERVAL_S="$STARTUP_POLL_INTERVAL_S" \
bash "$SCRIPT_DIR/start_native_topology.sh" "$HRM_MODEL" "$DEFAULT_LLM_MODEL"

router_flags=(
  --host "$ROUTER_HOST"
  --port "$ROUTER_PORT"
  --endpoint_solo_22gb "http://127.0.0.1:$PORT_SOLO_22GB"
  --endpoint_nvlink_pair "http://127.0.0.1:$PORT_NVLINK_PAIR"
  --endpoint_solo_3080 "http://127.0.0.1:$PORT_SOLO_3080"
  --short_prompt_tokens "$ROUTER_SHORT_PROMPT_TOKENS"
  --medium_prompt_tokens "$ROUTER_MEDIUM_PROMPT_TOKENS"
  --short_max_new_tokens "$ROUTER_SHORT_MAX_NEW_TOKENS"
  --medium_max_new_tokens "$ROUTER_MEDIUM_MAX_NEW_TOKENS"
  --long_max_new_tokens "$ROUTER_LONG_MAX_NEW_TOKENS"
  --request_timeout_s "$ROUTER_REQUEST_TIMEOUT_S"
  --health_timeout_s "$ROUTER_HEALTH_TIMEOUT_S"
  --max_concurrent "$ROUTER_MAX_CONCURRENT"
  --chars_per_token "$ROUTER_CHARS_PER_TOKEN"
  --default_mode "$ROUTER_DEFAULT_MODE"
)

if [[ -n "${ROUTER_TOKENIZER_MODEL:-}" ]]; then
  router_flags+=(--tokenizer_model "$ROUTER_TOKENIZER_MODEL")
fi
if [[ "$ROUTER_LOCAL_FILES_ONLY" == "1" ]]; then
  router_flags+=(--local_files_only)
fi
if [[ "$ROUTER_DISABLE_TOKENIZER" == "1" ]]; then
  router_flags+=(--disable_tokenizer)
fi

echo "[stack] starting router on $ROUTER_HOST:$ROUTER_PORT"
nohup "${HRM_FLASH_CMD[@]}" router "${router_flags[@]}" >"$LOG_DIR/router.log" 2>&1 &
echo $! > "$LOG_DIR/router.pid"

if command -v curl >/dev/null 2>&1; then
  router_url="http://127.0.0.1:${ROUTER_PORT}/v1/health"
  waited=0
  echo "[wait] router -> $router_url (timeout=${STARTUP_WAIT_TIMEOUT_S}s)"
  while (( waited < STARTUP_WAIT_TIMEOUT_S )); do
    if health_ok "$router_url"; then
      echo "[ready] router"
      break
    fi
    sleep "$STARTUP_POLL_INTERVAL_S"
    waited=$((waited + STARTUP_POLL_INTERVAL_S))
  done
  if (( waited >= STARTUP_WAIT_TIMEOUT_S )); then
    echo "ERR: router did not become healthy within ${STARTUP_WAIT_TIMEOUT_S}s" >&2
    if [[ -f "$LOG_DIR/router.log" ]]; then
      echo "----- last 80 lines: $LOG_DIR/router.log -----" >&2
      tail -n 80 "$LOG_DIR/router.log" >&2 || true
      echo "----------------------------------------------" >&2
    fi
    exit 1
  fi
else
  echo "[warn] curl not found; skipping router health wait."
fi

cat <<EOF

Native stack started.
Logs: $LOG_DIR

Services:
  solo_22gb   -> http://127.0.0.1:$PORT_SOLO_22GB/v1/health
  nvlink_pair -> http://127.0.0.1:$PORT_NVLINK_PAIR/v1/health
  solo_3080   -> http://127.0.0.1:$PORT_SOLO_3080/v1/health$(
if [[ "$ENABLE_SOLO_3080" == "0" ]]; then
  printf ' (alias to max-model lane)'
fi
)
  router      -> http://127.0.0.1:$ROUTER_PORT/v1/health

Inference entrypoint:
  POST http://127.0.0.1:$ROUTER_PORT/v1/generate

EOF
