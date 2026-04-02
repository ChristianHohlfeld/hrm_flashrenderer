#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <modeldir> <prompt>" >&2
  exit 2
fi
MODEL="$1"
PROMPT="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HRM_BIN="$ROOT_DIR/hrm_core/build/hrm"
[[ -x "$HRM_BIN" ]] || { echo "ERR: missing HRM binary: $HRM_BIN (run scripts/build.sh first)" >&2; exit 1; }
"$HRM_BIN" query --model "$MODEL" --format json --prompt "$PROMPT"
