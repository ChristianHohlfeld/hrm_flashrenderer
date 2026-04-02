#!/usr/bin/env bash
set -euo pipefail

# Full production E2E on real hardware:
# - optional bootstrap (build + deps)
# - preflight
# - start native stack
# - verify health
# - run final prompt through router and require real retrieval+inference
#
# Usage:
#   scripts/prod_live_e2e.sh <hrm_model_dir> [final_prompt]
#
# Environment knobs:
#   RUN_BOOTSTRAP=1|0         default: 1
#   EXPECTED_GPUS=<int>       default: 4
#   TOPOLOGY_MODE             default: max_model_fast
#   ROUTER_URL=<url>          default: http://127.0.0.1:8090
#   ROUTER_ROUTE_HINT         default: balanced
#   ROUTER_MAX_NEW_TOKENS     default: 256
#   AUTO_STOP=1|0             default: 0
#   ALLOW_EMPTY_SOURCES=1|0   default: 0 (0 = require non-empty sources => real pipeline)
#   PREFLIGHT_SCRIPT          default: scripts/prod_preflight.sh
#   START_STACK_SCRIPT        default: scripts/start_native_stack.sh
#   STOP_STACK_SCRIPT         default: scripts/stop_native_stack.sh
#   HEALTH_URL_SOLO_22GB      default: http://127.0.0.1:8081/v1/health
#   HEALTH_URL_NVLINK_PAIR    default:
#                             - max_model_fast: alias to solo_22gb URL
#                             - hetero_3lane: http://127.0.0.1:8082/v1/health
#   HEALTH_URL_SOLO_3080      default: http://127.0.0.1:8083/v1/health

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <hrm_model_dir> [final_prompt]" >&2
  exit 2
fi

HRM_MODEL="$1"
FINAL_PROMPT="${2:-Bitte antworte auf Deutsch in 3 kurzen Bulletpoints: 1) Stack-Status 2) Wichtigster Fakt aus den Quellen 3) Route-Hinweis.}"

RUN_BOOTSTRAP="${RUN_BOOTSTRAP:-1}"
EXPECTED_GPUS="${EXPECTED_GPUS:-4}"
TOPOLOGY_MODE="${TOPOLOGY_MODE:-max_model_fast}"
ROUTER_URL="${ROUTER_URL:-http://127.0.0.1:8090}"
ROUTER_ROUTE_HINT="${ROUTER_ROUTE_HINT:-balanced}"
ROUTER_MAX_NEW_TOKENS="${ROUTER_MAX_NEW_TOKENS:-256}"
AUTO_STOP="${AUTO_STOP:-0}"
ALLOW_EMPTY_SOURCES="${ALLOW_EMPTY_SOURCES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/.run/services}"
PREFLIGHT_SCRIPT="${PREFLIGHT_SCRIPT:-$SCRIPT_DIR/prod_preflight.sh}"
START_STACK_SCRIPT="${START_STACK_SCRIPT:-$SCRIPT_DIR/start_native_stack.sh}"
STOP_STACK_SCRIPT="${STOP_STACK_SCRIPT:-$SCRIPT_DIR/stop_native_stack.sh}"
PORT_SOLO_22GB="${PORT_SOLO_22GB:-8081}"
if [[ "$TOPOLOGY_MODE" == "max_model_fast" ]]; then
  PORT_NVLINK_PAIR="${PORT_NVLINK_PAIR:-$PORT_SOLO_22GB}"
else
  PORT_NVLINK_PAIR="${PORT_NVLINK_PAIR:-8082}"
fi
PORT_SOLO_3080="${PORT_SOLO_3080:-8083}"
HEALTH_URL_SOLO_22GB="${HEALTH_URL_SOLO_22GB:-http://127.0.0.1:${PORT_SOLO_22GB}/v1/health}"
HEALTH_URL_NVLINK_PAIR="${HEALTH_URL_NVLINK_PAIR:-http://127.0.0.1:${PORT_NVLINK_PAIR}/v1/health}"
HEALTH_URL_SOLO_3080="${HEALTH_URL_SOLO_3080:-http://127.0.0.1:${PORT_SOLO_3080}/v1/health}"

STACK_STARTED=0
cleanup() {
  if [[ "$AUTO_STOP" == "1" && "$STACK_STARTED" == "1" ]]; then
    echo "[cleanup] stopping native stack..."
    bash "$STOP_STACK_SCRIPT" || true
  fi
}
trap cleanup EXIT

require_cmd() {
  local c="$1"
  command -v "$c" >/dev/null 2>&1 || { echo "ERR: missing command: $c" >&2; exit 1; }
}

health_wait_ok() {
  local url="$1"
  local timeout_s="${2:-180}"
  local poll_s="${3:-2}"
  local waited=0
  while (( waited < timeout_s )); do
    local out
    out="$(curl -fsS --max-time 2 "$url" 2>/dev/null || true)"
    if [[ -n "$out" ]]; then
      if printf '%s' "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); raise SystemExit(0 if bool(d.get('ok')) else 1)" >/dev/null 2>&1; then
        return 0
      fi
    fi
    sleep "$poll_s"
    waited=$((waited + poll_s))
  done
  return 1
}

check_backend_deepseek() {
  local name="$1"
  local url="$2"
  local out
  out="$(curl -fsS "$url")"
  printf '%s' "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); \
ok=bool(d.get('ok')); b=d.get('backend'); r=bool(d.get('deepseek_running')); \
raise SystemExit(0 if ok and b=='deepseek_int8' and r else 1)" || {
    echo "ERR: backend health check failed for $name ($url)" >&2
    echo "$out" >&2
    exit 1
  }
  echo "[ok] $name deepseek_int8 running"
}

cd "$ROOT_DIR"
require_cmd bash
require_cmd curl
require_cmd python3
[[ -f "$PREFLIGHT_SCRIPT" ]] || { echo "ERR: missing preflight script: $PREFLIGHT_SCRIPT" >&2; exit 1; }
[[ -f "$START_STACK_SCRIPT" ]] || { echo "ERR: missing start script: $START_STACK_SCRIPT" >&2; exit 1; }
[[ -f "$STOP_STACK_SCRIPT" ]] || { echo "ERR: missing stop script: $STOP_STACK_SCRIPT" >&2; exit 1; }

echo "[1/5] bootstrap (RUN_BOOTSTRAP=$RUN_BOOTSTRAP)"
if [[ "$RUN_BOOTSTRAP" == "1" ]]; then
  require_cmd cmake
  require_cmd ctest
  PROFILE=deepseek_int8 bash "$SCRIPT_DIR/bootstrap.sh"
else
  echo "[skip] bootstrap skipped"
fi

echo "[2/5] production preflight"
bash "$PREFLIGHT_SCRIPT" "$HRM_MODEL" "$EXPECTED_GPUS"

echo "[3/5] start native stack"
export BACKEND=deepseek_int8
export PREPARE_MODELS=1
export TOPOLOGY_MODE="$TOPOLOGY_MODE"
bash "$START_STACK_SCRIPT" "$HRM_MODEL" auto
STACK_STARTED=1

echo "[4/5] verify service health"
health_wait_ok "$HEALTH_URL_SOLO_22GB" 180 2 || { echo "ERR: solo_22gb did not become healthy" >&2; exit 1; }
health_wait_ok "$HEALTH_URL_NVLINK_PAIR" 180 2 || { echo "ERR: nvlink_pair did not become healthy" >&2; exit 1; }
health_wait_ok "$HEALTH_URL_SOLO_3080" 180 2 || { echo "ERR: solo_3080 did not become healthy" >&2; exit 1; }
health_wait_ok "$ROUTER_URL/v1/health" 180 2 || { echo "ERR: router did not become healthy" >&2; exit 1; }

check_backend_deepseek "solo_22gb" "$HEALTH_URL_SOLO_22GB"
check_backend_deepseek "nvlink_pair" "$HEALTH_URL_NVLINK_PAIR"
check_backend_deepseek "solo_3080" "$HEALTH_URL_SOLO_3080"

echo "[5/5] final prompt through full router stack"
payload="$(
  FINAL_PROMPT="$FINAL_PROMPT" ROUTER_ROUTE_HINT="$ROUTER_ROUTE_HINT" ROUTER_MAX_NEW_TOKENS="$ROUTER_MAX_NEW_TOKENS" \
  python3 - <<'PY'
import json, os
print(json.dumps({
    "prompt": os.environ["FINAL_PROMPT"],
    "route_hint": os.environ["ROUTER_ROUTE_HINT"],
    "max_new_tokens": int(os.environ["ROUTER_MAX_NEW_TOKENS"]),
    "allow_failover": True,
}))
PY
)"

resp="$(curl -fsS -X POST "$ROUTER_URL/v1/generate" -H 'Content-Type: application/json' -d "$payload")"
RESP_JSON="$resp" python3 - <<'PY'
import json, os
d = json.loads(os.environ["RESP_JSON"])
if not bool(d.get("ok", False)):
    raise SystemExit("router response ok=false")

txt = str(d.get("text", "")).strip()
if not txt:
    raise SystemExit("empty response text")

sources = d.get("sources", [])
allow_empty = os.environ.get("ALLOW_EMPTY_SOURCES", "0") == "1"
if not allow_empty and (not isinstance(sources, list) or len(sources) == 0):
    raise SystemExit(
        "sources are empty -> this likely hit retrieval fallback instead of full retrieval+deepseek inference. "
        "Use a prompt that matches your HRM index or set ALLOW_EMPTY_SOURCES=1 if intentional."
    )

route = d.get("route", {}) if isinstance(d.get("route"), dict) else {}
selected = route.get("selected", "?")
prompt_tokens = route.get("prompt_tokens", "?")
lat_ms = route.get("latency_ms", "?")
print("[ok] final prompt succeeded")
print(f"[route] selected={selected} prompt_tokens={prompt_tokens} latency_ms={lat_ms}")
print("[text]")
print(txt)
PY

echo
echo "E2E PASS: DeepSeek stack started and final router prompt returned."
if [[ "$AUTO_STOP" == "0" ]]; then
  echo "Stack is still running. Stop with: bash scripts/stop_native_stack.sh"
fi
