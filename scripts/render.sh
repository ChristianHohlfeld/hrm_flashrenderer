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
python renderer/hrm_render.py --model "$MODEL" --llm "$GGUF" --prompt "$PROMPT" --n_gpu_layers "$GPU_LAYERS" --json
