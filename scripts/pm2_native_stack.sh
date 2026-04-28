#!/usr/bin/env bash
set -euo pipefail

# Long-running PM2 wrapper for the native stack. It owns the service lock,
# starts the stack once, and keeps PM2 attached for restart-on-failure.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

HRM_MODEL="${HRM_MODEL:-./model_index}"
PROFILE="${PROFILE:-${1:-A}}"
MODEL_QUANT="${MODEL_QUANT:-${2:-q8}}"
ROUTER_PORT="${ROUTER_PORT:-8090}"
PORT_SOLO_22GB="${PORT_SOLO_22GB:-8081}"
LOCK_FILE="${LOCK_FILE:-/tmp/hrm_flashrenderer_native_stack.lock}"
HEALTH_INTERVAL_S="${HEALTH_INTERVAL_S:-20}"
HEALTH_FAIL_LIMIT="${HEALTH_FAIL_LIMIT:-3}"

if [[ -z "${PYTHON_BIN:-}" && -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi
export PYTHON_BIN="${PYTHON_BIN:-python3}"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "ERR: another hrm_flashrenderer native stack wrapper holds $LOCK_FILE" >&2
  exit 1
fi

cleanup() {
  set +e
  bash "$SCRIPT_DIR/stop_native_stack.sh"
}
trap cleanup EXIT INT TERM

kill_stale_runtime() {
  set +e
  bash "$SCRIPT_DIR/stop_native_stack.sh"
  pkill -9 -f "$ROOT_DIR/.run/bin/deepseek_engine"
  pkill -9 -f "python.*-m hrm_flash.cli (serve|router)"
  set -e
}

wait_ports_clear() {
  local deadline=$((SECONDS + 20))
  while ss -ltn "( sport = :$PORT_SOLO_22GB or sport = :$ROUTER_PORT )" | grep -q LISTEN; do
    if (( SECONDS >= deadline )); then
      echo "ERR: ports still busy after cleanup: $PORT_SOLO_22GB/$ROUTER_PORT" >&2
      ss -ltnp | grep -E ":($PORT_SOLO_22GB|$ROUTER_PORT)" >&2 || true
      exit 1
    fi
    sleep 1
  done
}

kill_stale_runtime
wait_ports_clear

bash "$SCRIPT_DIR/start_easy.sh" "$HRM_MODEL" "$PROFILE" "$MODEL_QUANT"

failures=0
while true; do
  if curl -fsS --max-time 5 "http://127.0.0.1:$ROUTER_PORT/v1/health" >/dev/null &&
     curl -fsS --max-time 5 "http://127.0.0.1:$PORT_SOLO_22GB/v1/health" >/dev/null; then
    failures=0
  else
    failures=$((failures + 1))
    echo "WARN: health check failed ($failures/$HEALTH_FAIL_LIMIT)" >&2
    if (( failures >= HEALTH_FAIL_LIMIT )); then
      echo "ERR: health failed repeatedly; exiting for PM2 restart" >&2
      exit 1
    fi
  fi
  sleep "$HEALTH_INTERVAL_S"
done
