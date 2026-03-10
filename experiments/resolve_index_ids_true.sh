#!/usr/bin/env bash
set -euo pipefail

INDEX="${INDEX:-./index_v7_k18192_k28192.bin}"
PY="${PY:-python3}"

if [[ ! -f "$INDEX" ]]; then
  echo "FATAL: index not found: $INDEX" >&2
  exit 1
fi

"$PY" - "$INDEX" "$@" <<'PY'
import re, sys, struct
from pathlib import Path

BASE_V = 256

class PairIndex:
    def __init__(self):
        self.k1 = 0
        self.k2 = 0
        self.pow2 = 0
        self.id2pair = []
        self.id2pair2 = []
        self.hkeys = []
        self.hvals = []

def load_index_v7(path: Path) -> PairIndex:
    b = path.read_bytes()
    if len(b) < 24:
        raise SystemExit("index too small")

    magic, ver, k1, k2, pow2, res = struct.unpack_from("<6I", b, 0)
    print(f"[index] magic=0x{magic:08x} ver={ver} k1={k1} k2={k2} pow2={pow2} res={res}")

    if magic != 0x37584449:
        raise SystemExit(f"bad magic: 0x{magic:08x}")
    if ver != 1:
        raise SystemExit(f"unexpected version: {ver}")
    if pow2 > 30:
        raise SystemExit(f"suspicious pow2={pow2}")

    off = 24
    pi = PairIndex()
    pi.k1 = k1
    pi.k2 = k2
    pi.pow2 = pow2

    need = off + k1 * 2 + k2 * 4
    if len(b) < need:
        raise SystemExit("truncated before id2pair/id2pair2")

    # use memoryview slices to avoid giant struct format strings
    mv = memoryview(b)

    id2pair_raw = mv[off : off + k1 * 2]
    pi.id2pair = [struct.unpack_from("<H", id2pair_raw, i * 2)[0] for i in range(k1)]
    off += k1 * 2

    id2pair2_raw = mv[off : off + k2 * 4]
    pi.id2pair2 = [struct.unpack_from("<I", id2pair2_raw, i * 4)[0] for i in range(k2)]
    off += k2 * 4

    table_size = 1 << pow2
    need_tail = table_size * 4 + table_size * 2
    if len(b) < off + need_tail:
        raise SystemExit("truncated before hash tables")

    hkeys_raw = mv[off : off + table_size * 4]
    pi.hkeys = [struct.unpack_from("<I", hkeys_raw, i * 4)[0] for i in range(table_size)]
    off += table_size * 4

    hvals_raw = mv[off : off + table_size * 2]
    pi.hvals = [struct.unpack_from("<H", hvals_raw, i * 2)[0] for i in range(table_size)]
    off += table_size * 2

    return pi

def decode_id(pi: PairIndex, idx: int, out: bytearray, depth=0):
    if depth > 64:
        return
    if idx < BASE_V:
        out.append(idx & 0xFF)
        return

    x = idx - BASE_V
    if x < 0:
        return

    if x < pi.k1:
        p = pi.id2pair[x]
        out.append(p & 0xFF)
        out.append((p >> 8) & 0xFF)
        return

    x -= pi.k1
    if x < 0 or x >= pi.k2:
        out.extend(f"<pair:{idx}>".encode("utf-8", errors="replace"))
        return

    k = pi.id2pair2[x]
    a = (k >> 16) & 0xFFFF
    b = k & 0xFFFF
    decode_id(pi, a, out, depth + 1)
    decode_id(pi, b, out, depth + 1)

def decode_ids(pi: PairIndex, ids):
    out = bytearray()
    for i in ids:
        decode_id(pi, i, out, 0)
    return bytes(out)

def parse_nums(s: str):
    nums = [int(x) for x in re.findall(r"<(\d+)>", s)]
    if nums:
        return nums
    return [int(x) for x in re.findall(r"\d+", s)]

def score_text(text: str):
    printable = sum(ch.isprintable() for ch in text)
    letters = sum(ch.isalpha() for ch in text)
    weird = text.count("�") + text.count("<pair:")
    return printable + letters * 2 - weird * 5

index_path = Path(sys.argv[1])
pi = load_index_v7(index_path)

args = sys.argv[2:]
lines = [" ".join(args)] if args else [ln.rstrip("\n") for ln in sys.stdin]

for line in lines:
    if not line.strip():
        continue
    ids = parse_nums(line)
    raw = decode_ids(pi, ids)
    text = raw.decode("utf-8", errors="replace")
    print("INPUT:", line)
    print("IDS:", ids)
    print("RAW_HEX:", raw.hex())
    print("TEXT:", text)
    print("SCORE:", score_text(text))
    print()
PY
