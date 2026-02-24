#!/usr/bin/env bash
set -euo pipefail

HRM_MODEL="${1:-./model}"
LLM_MODEL="${2:-}"
WORLD="${3:-2}"
PORT="${4:-8080}"

if [[ -z "$LLM_MODEL" ]]; then
  echo "Usage: $0 <hrm_model_dir> <llm_model_dir> [world] [port]" >&2
  exit 2
fi

python -m pip install -U fastapi uvicorn pydantic

hrm-flash serve \
  --hrm_model "$HRM_MODEL" \
  --llm_model "$LLM_MODEL" \
  --world "$WORLD" \
  --port "$PORT" \
  --local_files_only
