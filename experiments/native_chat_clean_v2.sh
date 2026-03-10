#!/usr/bin/env bash
set -euo pipefail

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
    CKPT="$ckpt" GPU="$GPU" "$RUNTIME_SH" 2>&1
}

resolve_raw() {
  local raw="$1"
  if [[ -z "$raw" ]]; then
    echo ""
    return 0
  fi
  "$RESOLVE_SH" "$raw" | awk '
    /^TEXT:/ { sub(/^TEXT:[[:space:]]*/, "", $0); print; found=1 }
    END { if (!found) exit 0 }
  '
}

show_ask() {
  local prompt="$1"
  local ckpt full raw txt
  ckpt="$(best_ckpt)"
  full="$(run_runtime_once "$prompt" "$ckpt" || true)"
  raw="$(printf "%s" "$full" | extract_raw_ids)"
  txt="$(resolve_raw "$raw")"

  echo "PROMPT: $prompt"
  echo "CKPT:   $ckpt"
  if [[ -n "$raw" ]]; then
    echo "RAW:    $raw"
    echo "TEXT:   $txt"
  else
    echo "RAW:    "
    echo "TEXT:   "
    echo "TRACE:"
    printf "%s\n" "$full" | tail -n 20
  fi
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
    local full raw txt
    full="$(run_runtime_once "$prompt" "$ckpt" || true)"
    raw="$(printf "%s" "$full" | extract_raw_ids)"
    txt="$(resolve_raw "$raw")"
    if [[ -n "$raw" ]]; then
      echo "RAW:  $raw"
      echo "TEXT: $txt"
    else
      echo "RAW:  "
      echo "TEXT: "
      echo "TRACE:"
      printf "%s\n" "$full" | tail -n 20
    fi
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
native_chat_clean_v2.sh

Commands:
  train-smoke
  train
  ask "prompt"
  chat
  resolve "<ids>"
  best-ckpt
  help
EOF
}

case "$cmd" in
  train-smoke) train_smoke ;;
  train) train_full ;;
  ask)
    [[ $# -ge 1 ]] || { echo "need prompt" >&2; exit 1; }
    show_ask "$1"
    ;;
  chat) chat_loop ;;
  resolve)
    [[ $# -ge 1 ]] || { echo "need raw ids like <123><456>" >&2; exit 1; }
    echo "RAW:  $1"
    echo -n "TEXT: "
    resolve_raw "$1"
    ;;
  best-ckpt) best_ckpt ;;
  help|-h|--help) usage ;;
  *)
    echo "unknown command: $cmd" >&2
    usage >&2
    exit 1
    ;;
esac

