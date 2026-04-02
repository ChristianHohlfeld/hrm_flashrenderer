#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d "${ROOT_DIR}/.run/test_benchmark_deepseek.XXXXXX")"
OUT_DIR="$TMP_DIR/out"

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

python3 "$ROOT_DIR/scripts/mock_stack_server.py" \
  --solo-port "$SOLO_PORT" \
  --nvlink-port "$NVLINK_PORT" \
  --solo3080-port "$SOLO3080_PORT" \
  --router-port "$ROUTER_PORT" \
  >"$TMP_DIR/mock_stack.log" 2>&1 &
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

echo "[test] benchmark_deepseek writes comparable outputs"
BENCH_MODES="mixed,deepseek_only" \
BENCH_ROUTE_HINTS="balanced" \
BENCH_SCENARIOS="short" \
BENCH_REPEATS=2 \
BENCH_WARMUP=0 \
BENCH_MAX_NEW_TOKENS=32 \
BENCH_TIMEOUT_S=30 \
BENCH_TOKENIZER_MODEL=off \
bash "$ROOT_DIR/scripts/benchmark_deepseek.sh" "http://127.0.0.1:$ROUTER_PORT" "$OUT_DIR"

[[ -f "$OUT_DIR/raw.csv" ]] || { echo "ERR: missing raw.csv" >&2; exit 1; }
[[ -f "$OUT_DIR/summary.csv" ]] || { echo "ERR: missing summary.csv" >&2; exit 1; }
[[ -f "$OUT_DIR/summary.txt" ]] || { echo "ERR: missing summary.txt" >&2; exit 1; }

rows_raw="$(tail -n +2 "$OUT_DIR/raw.csv" | wc -l | tr -d ' ')"
if [[ "$rows_raw" -lt 4 ]]; then
  echo "ERR: expected at least 4 benchmark rows, got $rows_raw" >&2
  exit 1
fi

grep -q "scenario,mode,route_hint" "$OUT_DIR/summary.csv"
grep -q "mixed,balanced" "$OUT_DIR/summary.csv"
grep -q "deepseek_only,balanced" "$OUT_DIR/summary.csv"
grep -q "nvlink_pair" "$OUT_DIR/summary.csv"

echo "[ok] benchmark_deepseek outputs are present and comparable"

