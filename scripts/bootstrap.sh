#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${PROFILE:-deepseek_int8}"  # deepseek-only production mainline
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$PROFILE" != "deepseek_int8" ]]; then
  echo "ERR: PROFILE=$PROFILE is not supported in production mainline. Use PROFILE=deepseek_int8." >&2
  exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "ERR: python3/python not found. Install Python 3.10-3.12." >&2
    exit 1
  fi
fi

echo "[0/4] Validate Python runtime ($PYTHON_BIN)"
"$PYTHON_BIN" - <<'PY'
import sys
v = sys.version_info
ok = (v.major == 3 and 10 <= v.minor <= 12)
print(f"python={v.major}.{v.minor}.{v.micro}")
if not ok:
    raise SystemExit("unsupported Python version for this stack; use Python 3.10-3.12")
PY

echo "[1/4] Build HRM core (Release)"
cmake -S "$ROOT/hrm_core" -B "$ROOT/hrm_core/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/hrm_core/build" -j

echo "[2/4] Run HRM unit tests"
ctest --test-dir "$ROOT/hrm_core/build"

echo "[3/4] Install Python dependencies (profile=$PROFILE)"
"$PYTHON_BIN" -m pip install -r "$ROOT/requirements.prod.txt"
"$PYTHON_BIN" -m pip install -r "$ROOT/requirements.server.txt"
"$PYTHON_BIN" -m pip install -e "$ROOT"

echo "[4/4] Native DeepSeek profile ready (torch path intentionally disabled in mainline)."

echo "OK"
