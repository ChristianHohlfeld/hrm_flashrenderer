#!/usr/bin/env bash
set -euo pipefail

# resolve_index_ids.sh
# Decode llm_engine_full_blast outputs like:
#   <12390><3447><7700>
# back through index_v7_* into a best-effort text expansion.

INDEX="${INDEX:-./index_v7_k18192_k28192.bin}"
PAIR_K1="${PAIR_K1:-8192}"
BASE_V="${BASE_V:-256}"
PY="${PY:-python3}"

if [[ ! -f "$INDEX" ]]; then
  echo "FATAL: index not found: $INDEX" >&2
  exit 1
fi

"$PY" - "$INDEX" "$PAIR_K1" "$BASE_V" "$@" <<'PY'
import re, sys, struct
from pathlib import Path

index_path = Path(sys.argv[1])
pair_k1 = int(sys.argv[2])
base_v = int(sys.argv[3])

data = index_path.read_bytes()
if len(data) < 24:
    raise SystemExit("index too small")

pow2 = struct.unpack_from("<I", data, 16)[0]
off = 24
id2pair_len = pair_k1 * 2
id2pair2_len = pow2 * 4

if len(data) < off + id2pair_len + id2pair2_len:
    raise SystemExit("index truncated")

id2pair = data[off:off+id2pair_len]
off += id2pair_len
id2pair2 = data[off:off+id2pair2_len]

def decode_pair_id(pid: int, memo):
    if pid in memo:
        return memo[pid]
    if pid < 0:
        return ""
    if pid < pair_k1:
        a = id2pair[pid*2 + 0]
        b = id2pair[pid*2 + 1]
        out = bytes([a, b]).decode("utf-8", errors="replace")
        memo[pid] = out
        return out
    rid = pid - pair_k1
    if rid < 0 or rid >= pow2:
        out = f"<pair:{pid}>"
        memo[pid] = out
        return out
    a, b = struct.unpack_from("<HH", id2pair2, rid * 4)
    sa = decode_symbol(a, memo)
    sb = decode_symbol(b, memo)
    out = sa + sb
    memo[pid] = out
    return out

def decode_symbol(sym: int, memo):
    if 0 <= sym < 256:
        try:
            return bytes([sym]).decode("utf-8", errors="replace")
        except Exception:
            return chr(sym)
    return decode_pair_id(sym - base_v, memo)

def decode_ids(ids):
    memo = {}
    pieces = [decode_symbol(i, memo) for i in ids]
    return "".join(pieces)

def parse_line(s: str):
    nums = [int(x) for x in re.findall(r"<(\d+)>", s)]
    if nums:
        return nums
    nums = [int(x) for x in re.findall(r"\d+", s)]
    return nums

args = sys.argv[4:]
if args:
    lines = [" ".join(args)]
else:
    lines = [ln.rstrip("\n") for ln in sys.stdin]

for line in lines:
    if not line.strip():
        continue
    ids = parse_line(line)
    if not ids:
        print("")
        continue
    text = decode_ids(ids)
    print(text)
PY

