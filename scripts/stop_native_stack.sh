#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/.run/services}"

stop_pid_file() {
  local f="$1"
  if [[ -f "$f" ]]; then
    local pid
    pid="$(cat "$f" || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "[stop] pid=$pid ($f)"
      kill "$pid" || true
    fi
    rm -f "$f"
  fi
}

stop_pid_file "$LOG_DIR/router.pid"
bash "$SCRIPT_DIR/stop_native_topology.sh"

echo "Stopped native stack (router + topology services, if running)."
