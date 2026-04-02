#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <modeldir> <gguf> <prompt> [n_gpu_layers]" >&2
  exit 2
fi
MODEL="$1"
GGUF="$2"
PROMPT="$3"
GPU_LAYERS="${4:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "ERR: python3/python not found" >&2
    exit 1
  fi
fi
"$PYTHON_BIN" "$ROOT_DIR/renderer/hrm_render.py" --model "$MODEL" --llm "$GGUF" --prompt "$PROMPT" --n_gpu_layers "$GPU_LAYERS" --json
