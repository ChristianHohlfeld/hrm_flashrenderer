#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input.txt> <outdir> [cluster_size]" >&2
  exit 2
fi
IN="$1"
OUT="$2"
CLUSTER="${3:-200}"
./hrm_core/build/hrm prep --input "$IN" --out payloads.jsonl --cluster-size "$CLUSTER"
./hrm_core/build/hrm build --payloads payloads.jsonl --outdir "$OUT"
echo "OK: model built at $OUT"
