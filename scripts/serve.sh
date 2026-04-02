#!/usr/bin/env bash
set -euo pipefail

# Streamlined serve helper for native backend.
#
# Usage:
#   scripts/serve.sh <hrm_model_dir> <llm_model_or_repo> [port]
#
# Optional env vars:
#   BACKEND (default: deepseek_int8)
#   WORLD (default: 1)
#   MODEL_BIN
#   TOKENIZER_MODEL
#   NATIVE_ENGINE_BIN
#   LOCAL_FILES_ONLY (default: 1)
#   MAX_NEW_TOKENS (default: 256)
#   MAX_SEQ_LEN (default: 4096)
#   PREFILL_CHUNK_SIZE (default: 512)
#   PYTHON_BIN (default: python3, fallback: python)
#   HRM_FLASH_BIN (optional: explicit hrm-flash executable override)

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <hrm_model_dir> <llm_model_or_repo> [port]" >&2
  exit 2
fi

HRM_MODEL="$1"
LLM_MODEL="$2"
PORT="${3:-8080}"

BACKEND="${BACKEND:-deepseek_int8}"
WORLD="${WORLD:-1}"
MODEL_BIN="${MODEL_BIN:-}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-}"
NATIVE_ENGINE_BIN="${NATIVE_ENGINE_BIN:-}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-512}"

if [[ "$BACKEND" != "deepseek_int8" ]]; then
  echo "ERR: BACKEND=$BACKEND is not supported in production mainline. Use deepseek_int8." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "ERR: python3/python not found. Install Python 3.10-3.12." >&2
    exit 1
  fi
fi
if [[ -n "${HRM_FLASH_BIN:-}" ]]; then
  HRM_FLASH_CMD=("$HRM_FLASH_BIN")
else
  export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
  HRM_FLASH_CMD=("$PYTHON_BIN" -m hrm_flash.cli)
fi

args=(
  --hrm_model "$HRM_MODEL"
  --llm_model "$LLM_MODEL"
  --world "$WORLD"
  --port "$PORT"
  --backend "$BACKEND"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --max_seq_len "$MAX_SEQ_LEN"
  --prefill_chunk_size "$PREFILL_CHUNK_SIZE"
)

if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  args+=(--local_files_only)
fi
if [[ -n "$MODEL_BIN" ]]; then
  args+=(--model_bin "$MODEL_BIN")
fi
if [[ -n "$TOKENIZER_MODEL" ]]; then
  args+=(--tokenizer_model "$TOKENIZER_MODEL")
fi
if [[ -n "$NATIVE_ENGINE_BIN" ]]; then
  args+=(--native_engine_bin "$NATIVE_ENGINE_BIN")
fi

"${HRM_FLASH_CMD[@]}" serve "${args[@]}"
