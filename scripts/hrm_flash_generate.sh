#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <hrm_model_dir> <llm_model_dir> <world:2|3|4> <prompt...>" >&2
  exit 2
fi

HRM_MODEL="$1"; shift
LLM_MODEL="$1"; shift
WORLD="$1"; shift
PROMPT="$*"

hrm-flash generate \
  --hrm_model "$HRM_MODEL" \
  --llm_model "$LLM_MODEL" \
  --world "$WORLD" \
  --prompt "$PROMPT" \
  --max_new_tokens 256 \
  --max_seq_len 8192 \
  --prefill_chunk_size 1024 \
  --local_files_only
