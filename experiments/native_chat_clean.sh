#!/usr/bin/env bash
set -euo pipefail

# native_chat_clean.sh
#
# Clean front-end for the bounded true-pair native stack.
#
# It wraps:
#   - native_truepair_complete_fixed.sh (preferred) or native_truepair_complete.sh
#   - llm_engine_full_blast.sh
#   - resolve_index_ids_true.sh
#
# Usage:
#   ./native_chat_clean.sh train-smoke
#   ./native_chat_clean.sh train
#   ./native_chat_clean.sh ask "hi"
#   ./native_chat_clean.sh chat
#   ./native_chat_clean.sh resolve '<9187><9187><10572>'
#   ./native_chat_clean.sh best-ckpt

WORKDIR="${WORKDIR:-$PWD}"
GPU="${GPU:-1}"
OUT_DIR="${OUT_DIR:-$WORKDIR/native_truepair_complete}"
TRAIN_SH="${TRAIN_SH:-}"
RUNTIME_SH="${RUNTIME_SH:-$WORKDIR/llm_engine_full_blast.sh}"
RESOLVE_SH="${RESOLVE_SH:-$WORKDIR/resolve_index_ids_true.sh}"

if [[ -z "$TRAIN_SH" ]]; then
  if [[ -f "$WORKDIR/native_truepair_complete_fixed.sh" ]]; then
    TRAIN_SH="$WORKDIR/native_truepair_complete_fixed.sh"
  else
    TRAIN_SH="$WORKDIR/native_truepair_complete.sh"
  fi
fi

need_file() {
  local f="$1"
  [[ -f "$f" ]] || { echo "FATAL: missing file: $f" >&2; exit 1; }
}

need_file "$TRAIN_SH"
need_file "$RUNTIME_SH"
need_file "$RESOLVE_SH"

cmd="${1:-help}"
shift || true

best_ckpt() {
  OUT_DIR="$OUT_DIR" python3 - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["OUT_DIR"]) / "selector_results.json"
if not p.exists():
    raise SystemExit(f"selector file not found: {p}")
j = json.loads(p.read_text(encoding="utf-8"))
print(j["best"]["ckpt"])
PY
}

extract_raw_ids() {
  python3 - <<'PY'
import re, sys
text = sys.stdin.read()
matches = re.findall(r'((?:<\d+>)+)', text)
print(matches[-1] if matches else "")
PY
}

run_runtime_once() {
  local prompt="$1"
  local ckpt="$2"
  printf "%s\n/quit\n" "$prompt" | env -u DMODEL -u NHEAD -u NLAY -u FFN -u TMAX \
    CKPT="$ckpt" GPU="$GPU" "$RUNTIME_SH" 2>/dev/null
}

resolve_raw() {
  local raw="$1"
  if [[ -z "$raw" ]]; then
    echo ""
    return 0
  fi
  "$RESOLVE_SH" "$raw" | awk '
    /^TEXT:/ { sub(/^TEXT:[[:space:]]*/, "", $0); print; exit }
  '
}

show_ask() {
  local prompt="$1"
  local ckpt raw txt
  ckpt="$(best_ckpt)"
  raw="$(run_runtime_once "$prompt" "$ckpt" | extract_raw_ids)"
  txt="$(resolve_raw "$raw")"
  echo "PROMPT: $prompt"
  echo "CKPT:   $ckpt"
  echo "RAW:    $raw"
  echo "TEXT:   $txt"
}

chat_loop() {
  local ckpt
  ckpt="$(best_ckpt)"
  echo "[*] Using checkpoint: $ckpt"
  echo "[*] Type /quit to exit"
  while true; do
    printf "> "
    IFS= read -r prompt || break
    [[ "$prompt" == "/quit" ]] && break
    local raw txt
    raw="$(run_runtime_once "$prompt" "$ckpt" | extract_raw_ids)"
    txt="$(resolve_raw "$raw")"
    echo "RAW:  $raw"
    echo "TEXT: $txt"
  done
}

train_smoke() {
  exec env GPU="$GPU" "$TRAIN_SH" --smoketest
}

train_full() {
  exec env GPU="$GPU" "$TRAIN_SH" --train
}

usage() {
  cat <<'EOF'
native_chat_clean.sh

Commands:
  train-smoke           Small smoke-test training run
  train                 Full training run using current env/defaults
  ask "prompt"          Ask one prompt and show RAW ids + resolved TEXT
  chat                  Interactive chat loop with automatic resolve
  resolve "<ids>"       Resolve raw PairIndex id output
  best-ckpt             Print currently selected best checkpoint
  help                  Show this help

Examples:
  ./native_chat_clean.sh train-smoke
  ./native_chat_clean.sh ask "hi"
  ./native_chat_clean.sh chat
  ./native_chat_clean.sh resolve '<9187><9187><10572><10572>'

Helpful env:
  GPU=1
  OUT_DIR=./native_truepair_complete
  TRAIN_SH=./native_truepair_complete_fixed.sh
  RUNTIME_SH=./llm_engine_full_blast.sh
  RESOLVE_SH=./resolve_index_ids_true.sh
EOF
}

case "$cmd" in
  train-smoke)
    train_smoke
    ;;
  train)
    train_full
    ;;
  ask)
    [[ $# -ge 1 ]] || { echo "need prompt" >&2; exit 1; }
    show_ask "$1"
    ;;
  chat)
    chat_loop
    ;;
  resolve)
    [[ $# -ge 1 ]] || { echo "need raw ids like <123><456>" >&2; exit 1; }
    echo "RAW:  $1"
    echo -n "TEXT: "
    resolve_raw "$1"
    ;;
  best-ckpt)
    best_ckpt
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "unknown command: $cmd" >&2
    usage >&2
    exit 1
    ;;
esac

