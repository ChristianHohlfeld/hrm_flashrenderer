#!/usr/bin/env bash
set -euo pipefail

# resolve_index_ids_diag.sh
# Tries multiple decode assumptions against index_v7 and prints candidates.

INDEX="${INDEX:-./index_v7_k18192_k28192.bin}"
PY="${PY:-python3}"

if [[ ! -f "$INDEX" ]]; then
  echo "FATAL: index not found: $INDEX" >&2
  exit 1
fi

"$PY" - "$INDEX" "$@" <<'PY'
import re, sys, struct
from pathlib import Path

index_path = Path(sys.argv[1])
args = sys.argv[2:]
data = index_path.read_bytes()

def parse_nums(s: str):
    nums = [int(x) for x in re.findall(r"<(\d+)>", s)]
    if nums: return nums
    return [int(x) for x in re.findall(r"\d+", s)]

lines = [" ".join(args)] if args else [ln.rstrip("\n") for ln in sys.stdin]

if len(data) < 24:
    raise SystemExit("index too small")

pow2 = struct.unpack_from("<I", data, 16)[0]
raw_tail = data[24:]

def try_decode(ids, pair_k1, base_v):
    id2pair_len = pair_k1 * 2
    id2pair2_len = pow2 * 4
    if len(raw_tail) < id2pair_len + id2pair2_len:
        return None, "tail too short"
    id2pair = raw_tail[:id2pair_len]
    id2pair2 = raw_tail[id2pair_len:id2pair_len+id2pair2_len]
    memo = {}

    def dec_sym(sym, depth=0):
        if depth > 32:
            return "<rec>"
        if 0 <= sym < 256:
            return bytes([sym]).decode("utf-8", errors="replace")
        pid = sym - base_v
        if pid in memo:
            return memo[pid]
        if pid < 0:
            return f"<neg:{sym}>"
        if pid < pair_k1:
            a = id2pair[pid*2 + 0]
            b = id2pair[pid*2 + 1]
            out = bytes([a,b]).decode("utf-8", errors="replace")
            memo[pid] = out
            return out
        rid = pid - pair_k1
        if rid < 0 or rid >= pow2:
            return f"<pair:{pid}>"
        a, b = struct.unpack_from("<HH", id2pair2, rid*4)
        out = dec_sym(a, depth+1) + dec_sym(b, depth+1)
        memo[pid] = out
        return out

    txt = "".join(dec_sym(i) for i in ids)

    printable = sum(ch.isprintable() for ch in txt)
    letters = sum(ch.isalpha() for ch in txt)
    weird = txt.count(" ") + txt.count("<pair:")
    score = printable + letters * 2 - weird * 5
    return (txt, score), None

pair_k1_options = [8192, 16384, 18192, 4096]
base_v_options = [256, 0, 1]

for line in lines:
    if not line.strip():
        continue
    ids = parse_nums(line)
    cands = []
    for pk in pair_k1_options:
        for bv in base_v_options:
            res, err = try_decode(ids, pk, bv)
            if res:
                txt, score = res
                cands.append((score, pk, bv, txt))
    cands.sort(reverse=True, key=lambda x: x[0])
    print("INPUT:", line)
    for score, pk, bv, txt in cands[:8]:
        print(f"[score={score:>4}] pair_k1={pk} base_v={bv} -> {txt}")
    print()
PY

