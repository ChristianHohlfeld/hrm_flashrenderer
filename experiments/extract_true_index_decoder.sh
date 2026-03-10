#!/usr/bin/env bash
set -euo pipefail

# extract_true_index_decoder.sh
# Scans the local repo for the real index/pair decode path and writes a report.

REPO_DIR="${REPO_DIR:-$PWD}"
OUT="${OUT:-$REPO_DIR/true_index_decode_report.txt}"

cd "$REPO_DIR"

echo "[*] repo: $REPO_DIR"
echo "[*] report: $OUT"

{
  echo "==== FILE CANDIDATES ===="
  find . -maxdepth 3 -type f \
    \( -name '*.cpp' -o -name '*.cu' -o -name '*.c' -o -name '*.h' -o -name '*.hpp' -o -name '*.sh' -o -name '*.py' \) \
    | grep -E 'index|pair|pho|adapter|llm_engine|run3|run_llm|deepseek|renderer' \
    | sort || true

  echo
  echo "==== GREP: id2pair / id2pair2 / pow2 / pair_k1 / k18192 / k28192 ===="
  grep -RInE 'id2pair2?|pow2|pair_k1|k18192|k28192|index_v7|pair index|PairIndex|PHO|pho' . \
    --include='*.cpp' --include='*.cu' --include='*.c' --include='*.h' --include='*.hpp' --include='*.sh' --include='*.py' || true

  echo
  echo "==== index_build_v7.cpp (head 260 lines) ===="
  if [[ -f ./experiments/index_build_v7.cpp ]]; then
    nl -ba ./experiments/index_build_v7.cpp | sed -n '1,260p'
  fi

  echo
  echo "==== index_build_v7.cpp (grep neighborhood) ===="
  if [[ -f ./experiments/index_build_v7.cpp ]]; then
    for pat in id2pair id2pair2 pow2 pair_k1 pair k18192 k28192; do
      echo "-- pattern: $pat --"
      grep -n "$pat" ./experiments/index_build_v7.cpp | cut -d: -f1 | while read -r ln; do
        [[ -n "$ln" ]] && sed -n "$((ln-8)),$((ln+20))p" ./experiments/index_build_v7.cpp
        echo
      done
    done
  fi

  echo
  echo "==== llm_engine.cu / generated engine decode hints ===="
  for f in ./experiments/llm_engine.cu ./experiments/llm_engine_mega.cu; do
    if [[ -f "$f" ]]; then
      echo "-- FILE: $f --"
      grep -nE 'id2pair2?|pow2|pair_k1|decode|render|token|pho|index_v7' "$f" | cut -d: -f1 | while read -r ln; do
        [[ -n "$ln" ]] && sed -n "$((ln-6)),$((ln+18))p" "$f"
        echo
      done
    fi
  done

  echo
  echo "==== adapter_engine.cpp / deepseek adapter hints ===="
  if [[ -f ./experiments/adapter_engine.cpp ]]; then
    grep -nE 'id2pair2?|pow2|pair_k1|decode|render|token|pho|index_v7' ./experiments/adapter_engine.cpp | cut -d: -f1 | while read -r ln; do
      [[ -n "$ln" ]] && sed -n "$((ln-6)),$((ln+18))p" ./experiments/adapter_engine.cpp
      echo
    done
  fi

  echo
  echo "==== run scripts mentioning index / pair / pho ===="
  for f in ./experiments/run_llm_orig.sh ./experiments/run_llm.sh ./experiments/run3.sh ./experiments/runbeast.sh ./experiments/deepseek_adapter.sh; do
    if [[ -f "$f" ]]; then
      echo "-- FILE: $f --"
      grep -nE 'index_v7|pair_k1|id2pair|pho|PHO|k18192|k28192' "$f" | cut -d: -f1 | while read -r ln; do
        [[ -n "$ln" ]] && sed -n "$((ln-4)),$((ln+12))p" "$f"
        echo
      done
    fi
  done

  echo
  echo "==== binary header peek ===="
  if [[ -f ./experiments/index_v7_k18192_k28192.bin ]]; then
    python3 - <<'PY'
from pathlib import Path
import struct
p = Path("./experiments/index_v7_k18192_k28192.bin")
b = p.read_bytes()
print("size", len(b))
print("first64", b[:64].hex())
if len(b) >= 24:
    print("u32[0:6]", struct.unpack("<6I", b[:24]))
PY
  fi
} > "$OUT"

echo "[*] wrote $OUT"
echo "[*] open the sections around index_build_v7.cpp and adapter/engine hints first"

