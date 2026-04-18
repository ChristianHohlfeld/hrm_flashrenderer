#!/usr/bin/env bash
set -euo pipefail

# One-command onboarding flow for first-time users.
#
# Default behavior:
# - ensures mandatory hardware selection file exists
# - creates a minimal local HRM index if none is provided
# - runs production preflight
# - starts native stack
# - waits for router health
#
# Usage:
#   scripts/first_user_start.sh [hrm_model_dir] [default_llm_model|auto]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_DIR="${1:-$ROOT_DIR/.run/first_user/model_index}"
DEFAULT_LLM_MODEL="${2:-auto}"
ROUTER_HOST="${ROUTER_HOST:-127.0.0.1}"
ROUTER_PORT="${ROUTER_PORT:-8090}"
HEALTH_URL="http://${ROUTER_HOST}:${ROUTER_PORT}/v1/health"

AUTO_GENERATE_MODEL_INDEX="${AUTO_GENERATE_MODEL_INDEX:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-120}"
HEALTH_POLL_S="${HEALTH_POLL_S:-2}"

usage() {
  cat <<EOF
Usage:
  $0 [hrm_model_dir] [default_llm_model|auto]

Examples:
  $0
  $0 ./model_index auto

Environment knobs:
  AUTO_GENERATE_MODEL_INDEX=1|0  (default: 1)
  RUN_PREFLIGHT=1|0              (default: 1)
  ROUTER_HOST                    (default: 127.0.0.1)
  ROUTER_PORT                    (default: 8090)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

need_cmd() {
  local c="$1"
  command -v "$c" >/dev/null 2>&1 || {
    echo "ERR: required command not found: $c" >&2
    exit 1
  }
}

has_index() {
  local d="$1"
  [[ -f "$d/router_index.bin" && -f "$d/index.sqlite" ]]
}

ensure_hw_selection() {
  local f="$ROOT_DIR/.hw_selection.env"
  if [[ -f "$f" ]]; then
    echo "[first-user] using existing hardware selection: $f"
    return
  fi
  echo "[first-user] no hardware selection found -> applying default pool"
  bash "$ROOT_DIR/scripts/hw_select.sh"
}

ensure_hrm_build() {
  if [[ -x "$ROOT_DIR/hrm_core/build/hrm" ]]; then
    return
  fi
  echo "[first-user] building HRM core (missing hrm_core/build/hrm)"
  bash "$ROOT_DIR/scripts/build.sh"
}

ensure_model_index() {
  if has_index "$MODEL_DIR"; then
    echo "[first-user] using model index: $MODEL_DIR"
    return
  fi

  if [[ "$AUTO_GENERATE_MODEL_INDEX" != "1" ]]; then
    echo "ERR: missing index in $MODEL_DIR and AUTO_GENERATE_MODEL_INDEX=0" >&2
    echo "Create index first or run with AUTO_GENERATE_MODEL_INDEX=1." >&2
    exit 1
  fi

  mkdir -p "$MODEL_DIR"
  ensure_hrm_build

  local seed_file="$ROOT_DIR/tests/fixtures/hrm_seed.txt"
  if [[ ! -f "$seed_file" ]]; then
    mkdir -p "$(dirname "$seed_file")"
    cat > "$seed_file" <<'EOF'
HRM onboarding seed corpus.
This file is used to generate a minimal local model index.
EOF
  fi

  echo "[first-user] generating minimal model index at: $MODEL_DIR"
  bash "$ROOT_DIR/scripts/make_model.sh" "$seed_file" "$MODEL_DIR" 8
}

wait_for_health() {
  local timeout="$HEALTH_TIMEOUT_S"
  local poll="$HEALTH_POLL_S"
  local waited=0
  while (( waited < timeout )); do
    if curl -fsS "$HEALTH_URL" >/tmp/hrm_first_user_health.json 2>/dev/null; then
      if python3 - <<'PY'
import json
from pathlib import Path
obj = json.loads(Path("/tmp/hrm_first_user_health.json").read_text(encoding="utf-8"))
raise SystemExit(0 if obj.get("ok") is True else 1)
PY
      then
        echo "[first-user] router healthy: $HEALTH_URL"
        rm -f /tmp/hrm_first_user_health.json
        return 0
      fi
    fi
    sleep "$poll"
    waited=$((waited + poll))
  done
  rm -f /tmp/hrm_first_user_health.json
  echo "ERR: router did not become healthy within ${timeout}s ($HEALTH_URL)" >&2
  return 1
}

echo "[first-user] root=$ROOT_DIR"
echo "[first-user] model_dir=$MODEL_DIR default_llm_model=$DEFAULT_LLM_MODEL"

need_cmd bash
need_cmd python3
need_cmd curl

ensure_hw_selection
ensure_model_index

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  echo "[first-user] running production preflight"
  bash "$ROOT_DIR/scripts/prod_preflight.sh" "$MODEL_DIR"
else
  echo "[first-user] skipping preflight (RUN_PREFLIGHT=0)"
fi

echo "[first-user] starting native stack"
bash "$ROOT_DIR/scripts/start_native_stack.sh" "$MODEL_DIR" "$DEFAULT_LLM_MODEL"

wait_for_health

cat <<EOF
[first-user] ready.

Try:
  curl -s ${HEALTH_URL}
  curl -s http://${ROUTER_HOST}:${ROUTER_PORT}/v1/generate -H 'Content-Type: application/json' -d '{"prompt":"Kurz antworten.","mode":"mixed"}'

Stop:
  bash $ROOT_DIR/scripts/stop_native_stack.sh
EOF
