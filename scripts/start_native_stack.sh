#!/usr/bin/env bash
set -euo pipefail

# Starts full native heterogeneous stack:
# 1) three hrm-flash serve services (22GB solo, 11+11 NVLink pair, 3080 solo)
# 2) one hrm-flash router endpoint with auto-dispatch + failover
#
# Usage:
#   scripts/start_native_stack.sh <hrm_model_dir> [default_llm_model|auto]
#
# Important env vars (optional):
#   BACKEND (default: deepseek_int8)
#   MODEL_PROFILE (default: max_vram_hetero for auto mode)
#   LLM_MODEL_SOLO_22GB / LLM_MODEL_NVLINK_PAIR / LLM_MODEL_SOLO_3080
#   STRICT_GPU_TOPOLOGY (default: 1, enforced by start_native_topology.sh)
#   PORT_SOLO_22GB, PORT_NVLINK_PAIR, PORT_SOLO_3080
#   ROUTER_HOST, ROUTER_PORT
#   ROUTER_MAX_CONCURRENT
#   ROUTER_SHORT_PROMPT_TOKENS, ROUTER_MEDIUM_PROMPT_TOKENS
#   ROUTER_SHORT_MAX_NEW_TOKENS, ROUTER_MEDIUM_MAX_NEW_TOKENS, ROUTER_LONG_MAX_NEW_TOKENS
#   ROUTER_REQUEST_TIMEOUT_S, ROUTER_HEALTH_TIMEOUT_S
#   ROUTER_TOKENIZER_MODEL
#   ROUTER_LOCAL_FILES_ONLY (default: LOCAL_FILES_ONLY or 1)
#   ROUTER_DISABLE_TOKENIZER (1 disables tokenizer-based routing)
#   PREPARE_MODELS (default: 1 for deepseek_int8 backend)
#   STARTUP_WAIT_TIMEOUT_S, STARTUP_POLL_INTERVAL_S (forwarded to topology and router wait)
#   LOG_DIR

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
PREPARE_MODELS="${PREPARE_MODELS:-1}"
MODEL_PROFILE="${MODEL_PROFILE:-max_vram_hetero}"
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

PORT_SOLO_22GB="${PORT_SOLO_22GB:-8081}"
PORT_NVLINK_PAIR="${PORT_NVLINK_PAIR:-8082}"
PORT_SOLO_3080="${PORT_SOLO_3080:-8083}"

RECO_MODEL_SOLO_22GB="${RECO_MODEL_SOLO_22GB:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}"
RECO_MODEL_NVLINK_PAIR="${RECO_MODEL_NVLINK_PAIR:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}"
RECO_MODEL_SOLO_3080="${RECO_MODEL_SOLO_3080:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"

use_profile=0
case "$DEFAULT_LLM_MODEL" in
  ""|auto|max|max_vram|max_vram_hetero|"<dein-deepseek-distill>")
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
  LLM_MODEL_SOLO_22GB="${LLM_MODEL_SOLO_22GB:-$RECO_MODEL_SOLO_22GB}"
  LLM_MODEL_NVLINK_PAIR="${LLM_MODEL_NVLINK_PAIR:-$RECO_MODEL_NVLINK_PAIR}"
  LLM_MODEL_SOLO_3080="${LLM_MODEL_SOLO_3080:-$RECO_MODEL_SOLO_3080}"
  DEFAULT_LLM_MODEL="$LLM_MODEL_SOLO_22GB"
else
  LLM_MODEL_SOLO_22GB="${LLM_MODEL_SOLO_22GB:-$DEFAULT_LLM_MODEL}"
  LLM_MODEL_NVLINK_PAIR="${LLM_MODEL_NVLINK_PAIR:-$DEFAULT_LLM_MODEL}"
  LLM_MODEL_SOLO_3080="${LLM_MODEL_SOLO_3080:-$DEFAULT_LLM_MODEL}"
fi

echo "[stack] backend=$BACKEND profile=$MODEL_PROFILE auto_profile=$use_profile"
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
ROUTER_LOCAL_FILES_ONLY="${ROUTER_LOCAL_FILES_ONLY:-${LOCAL_FILES_ONLY:-1}}"

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
      bash "$BUILD_SCRIPT" "$model" "$out_bin"
    else
      bash "$BUILD_SCRIPT" "$model"
    fi
  }
  build_one "${LLM_MODEL_SOLO_22GB:-$DEFAULT_LLM_MODEL}" "${MODEL_BIN_SOLO_22GB:-}"
  build_one "${LLM_MODEL_NVLINK_PAIR:-$DEFAULT_LLM_MODEL}" "${MODEL_BIN_NVLINK_PAIR:-}"
  build_one "${LLM_MODEL_SOLO_3080:-$DEFAULT_LLM_MODEL}" "${MODEL_BIN_SOLO_3080:-}"
fi

echo "[stack] starting native topology services..."
MODEL_PROFILE="$MODEL_PROFILE" \
STRICT_GPU_TOPOLOGY="$STRICT_GPU_TOPOLOGY" \
LLM_MODEL_SOLO_22GB="$LLM_MODEL_SOLO_22GB" \
LLM_MODEL_NVLINK_PAIR="$LLM_MODEL_NVLINK_PAIR" \
LLM_MODEL_SOLO_3080="$LLM_MODEL_SOLO_3080" \
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
nohup hrm-flash router "${router_flags[@]}" >"$LOG_DIR/router.log" 2>&1 &
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
  solo_3080   -> http://127.0.0.1:$PORT_SOLO_3080/v1/health
  router      -> http://127.0.0.1:$ROUTER_PORT/v1/health

Inference entrypoint:
  POST http://127.0.0.1:$ROUTER_PORT/v1/generate

EOF
