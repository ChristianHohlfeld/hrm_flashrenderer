#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <modeldir> <prompt>" >&2
  exit 2
fi
MODEL="$1"
PROMPT="$2"
./hrm_core/build/hrm query --model "$MODEL" --format json --prompt "$PROMPT"
