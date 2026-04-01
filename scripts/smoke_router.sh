#!/usr/bin/env bash
set -euo pipefail

# Quick real-world smoke test for router and backend routing decisions.
#
# Usage:
#   scripts/smoke_router.sh [router_base_url]
#
# Example:
#   scripts/smoke_router.sh http://127.0.0.1:8090

ROUTER_URL="${1:-http://127.0.0.1:8090}"
PROMPT="${PROMPT:-Summarize the key retrieval evidence in 5 concise bullets.}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"

if ! command -v curl >/dev/null 2>&1; then
  echo "ERR: curl is required." >&2
  exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERR: python3 is required for JSON parsing in this smoke test." >&2
  exit 2
fi

echo "[smoke] router health: $ROUTER_URL/v1/health"
curl -s "$ROUTER_URL/v1/health" | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps({'ok':d.get('ok'), 'backends':d.get('backends')}, indent=2))"

run_case() {
  local label="$1"
  local payload="$2"
  local t0 t1 latency resp selected p_tokens
  t0=$(date +%s%3N)
  resp="$(curl -s -X POST "$ROUTER_URL/v1/generate" -H 'Content-Type: application/json' -d "$payload")"
  t1=$(date +%s%3N)
  latency=$((t1 - t0))

  selected="$(printf '%s' "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('route',{}).get('selected','?'))")"
  p_tokens="$(printf '%s' "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('route',{}).get('prompt_tokens','?'))")"
  echo "[smoke] $label -> selected=$selected prompt_tokens=$p_tokens latency_ms=$latency"
}

payload_auto="$(PROMPT="$PROMPT" MAX_NEW_TOKENS="$MAX_NEW_TOKENS" python3 -c "import json,os; print(json.dumps({'prompt': os.environ['PROMPT'], 'max_new_tokens': int(os.environ['MAX_NEW_TOKENS'])}))")"
payload_fast="$(PROMPT="$PROMPT" MAX_NEW_TOKENS="$MAX_NEW_TOKENS" python3 -c "import json,os; print(json.dumps({'prompt': os.environ['PROMPT'], 'max_new_tokens': int(os.environ['MAX_NEW_TOKENS']), 'route_hint': 'fast'}))")"
payload_quality="$(PROMPT="$PROMPT" MAX_NEW_TOKENS="$MAX_NEW_TOKENS" python3 -c "import json,os; print(json.dumps({'prompt': os.environ['PROMPT'], 'max_new_tokens': int(os.environ['MAX_NEW_TOKENS']), 'route_hint': 'quality'}))")"

run_case "auto" "$payload_auto"
run_case "fast" "$payload_fast"
run_case "quality" "$payload_quality"

echo "[smoke] done"
