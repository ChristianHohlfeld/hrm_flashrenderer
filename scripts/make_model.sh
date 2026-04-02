#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input.txt> <outdir> [cluster_size]" >&2
  exit 2
fi
IN="$1"
OUT="$2"
CLUSTER="${3:-200}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HRM_BIN="$ROOT_DIR/hrm_core/build/hrm"
[[ -x "$HRM_BIN" ]] || { echo "ERR: missing HRM binary: $HRM_BIN (run scripts/build.sh first)" >&2; exit 1; }

"$HRM_BIN" prep --input "$IN" --out "$ROOT_DIR/payloads.jsonl" --cluster-size "$CLUSTER"
"$HRM_BIN" build --payloads "$ROOT_DIR/payloads.jsonl" --outdir "$OUT"
echo "OK: model built at $OUT"
