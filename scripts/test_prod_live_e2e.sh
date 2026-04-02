#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d "${ROOT_DIR}/.run/test_prod_live_e2e.XXXXXX")"
MODEL_DIR="$TMP_DIR/model"
LOG_DIR="$TMP_DIR/logs"
mkdir -p "$MODEL_DIR" "$LOG_DIR"
touch "$MODEL_DIR/router_index.bin" "$MODEL_DIR/index.sqlite"

cleanup() {
  if [[ -n "${MOCK_PID:-}" ]] && kill -0 "$MOCK_PID" 2>/dev/null; then
    kill "$MOCK_PID" >/dev/null 2>&1 || true
    wait "$MOCK_PID" 2>/dev/null || true
  fi
  rm -rf "$TMP_DIR" || true
}
trap cleanup EXIT

read -r SOLO_PORT NVLINK_PORT SOLO3080_PORT ROUTER_PORT <<< "$(python3 - <<'PY'
import socket
ports = []
for _ in range(4):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    ports.append(str(s.getsockname()[1]))
    s.close()
print(" ".join(ports))
PY
)"

cat > "$TMP_DIR/preflight_ok.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 2 ]] || { echo "need args" >&2; exit 1; }
[[ -f "$1/router_index.bin" ]] || { echo "missing router_index.bin" >&2; exit 1; }
[[ -f "$1/index.sqlite" ]] || { echo "missing index.sqlite" >&2; exit 1; }
[[ "$2" =~ ^[0-9]+$ ]] || { echo "expected gpu count not numeric" >&2; exit 1; }
EOF
chmod +x "$TMP_DIR/preflight_ok.sh"

cat > "$TMP_DIR/start_ok.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ "${BACKEND:-}" == "deepseek_int8" ]] || { echo "BACKEND not deepseek_int8" >&2; exit 1; }
[[ "${PREPARE_MODELS:-}" == "1" ]] || { echo "PREPARE_MODELS not 1" >&2; exit 1; }
exit 0
EOF
chmod +x "$TMP_DIR/start_ok.sh"

cat > "$TMP_DIR/stop_ok.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exit 0
EOF
chmod +x "$TMP_DIR/stop_ok.sh"

start_mock() {
  local extra="$1"
  if [[ -n "${MOCK_PID:-}" ]] && kill -0 "$MOCK_PID" 2>/dev/null; then
    kill "$MOCK_PID" >/dev/null 2>&1 || true
    wait "$MOCK_PID" 2>/dev/null || true
  fi

  python3 "$ROOT_DIR/scripts/mock_stack_server.py" \
    --solo-port "$SOLO_PORT" \
    --nvlink-port "$NVLINK_PORT" \
    --solo3080-port "$SOLO3080_PORT" \
    --router-port "$ROUTER_PORT" \
    $extra >"$LOG_DIR/mock_stack.log" 2>&1 &
  MOCK_PID=$!

  for u in \
    "http://127.0.0.1:$SOLO_PORT/v1/health" \
    "http://127.0.0.1:$NVLINK_PORT/v1/health" \
    "http://127.0.0.1:$SOLO3080_PORT/v1/health" \
    "http://127.0.0.1:$ROUTER_PORT/v1/health"; do
    for _ in $(seq 1 40); do
      if curl -fsS "$u" >/dev/null 2>&1; then
        break
      fi
      sleep 0.1
    done
  done
}

run_e2e() {
  local allow_empty="$1"
  local prompt="$2"
  RUN_BOOTSTRAP=0 \
  EXPECTED_GPUS=4 \
  AUTO_STOP=1 \
  ALLOW_EMPTY_SOURCES="$allow_empty" \
  PREFLIGHT_SCRIPT="$TMP_DIR/preflight_ok.sh" \
  START_STACK_SCRIPT="$TMP_DIR/start_ok.sh" \
  STOP_STACK_SCRIPT="$TMP_DIR/stop_ok.sh" \
  HEALTH_URL_SOLO_22GB="http://127.0.0.1:$SOLO_PORT/v1/health" \
  HEALTH_URL_NVLINK_PAIR="http://127.0.0.1:$NVLINK_PORT/v1/health" \
  HEALTH_URL_SOLO_3080="http://127.0.0.1:$SOLO3080_PORT/v1/health" \
  ROUTER_URL="http://127.0.0.1:$ROUTER_PORT" \
  bash "$ROOT_DIR/scripts/prod_live_e2e.sh" "$MODEL_DIR" "$prompt"
}

echo "[test] prod_live_e2e success path with sources"
start_mock ""
run_e2e 0 "Nenne den wichtigsten Punkt."

echo "[test] prod_live_e2e fails hard when sources are empty"
start_mock "--empty-sources"
set +e
out_empty="$(run_e2e 0 "Leere Quellen testen." 2>&1)"
code_empty=$?
set -e
if [[ "$code_empty" -eq 0 ]]; then
  echo "ERR: prod_live_e2e unexpectedly succeeded with empty sources" >&2
  exit 1
fi
printf '%s' "$out_empty" | grep -Eq "sources are empty|source_count is 0"

echo "[test] prod_live_e2e can allow empty sources when explicitly enabled"
run_e2e 1 "Leere Quellen mit Allow."

echo "[ok] prod_live_e2e hard checks"
