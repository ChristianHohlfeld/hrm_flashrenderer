#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-./.run/services}"

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

stop_pid_file "$LOG_DIR/solo_22gb.pid"
stop_pid_file "$LOG_DIR/nvlink_pair.pid"
stop_pid_file "$LOG_DIR/solo_3080.pid"

echo "Stopped topology services (if running)."
