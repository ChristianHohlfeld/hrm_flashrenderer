import os, re, struct, sys
from pathlib import Path
BASE_V = 256
class PairIndex:
    def __init__(self):
        self.k1=0; self.k2=0; self.pow2=0; self.id2pair=[]; self.id2pair2=[]
def load_index_v7(path: Path):
    b = path.read_bytes()
    magic, ver, k1, k2, pow2, res = struct.unpack_from("<6I", b, 0)
    off=24; mv=memoryview(b); pi=PairIndex(); pi.k1=k1; pi.k2=k2; pi.pow2=pow2
    raw = mv[off:off+k1*2]; pi.id2pair=[struct.unpack_from("<H", raw, i*2)[0] for i in range(k1)]; off += k1*2
    raw = mv[off:off+k2*4]; pi.id2pair2=[struct.unpack_from("<I", raw, i*4)[0] for i in range(k2)]; return pi
def decode_id(pi, idx, out, depth=0):
    if depth>64: return
    if idx<BASE_V: out.append(idx&0xff); return
    x=idx-BASE_V
    if x<pi.k1:
        p=pi.id2pair[x]; out.append(p&0xff); out.append((p>>8)&0xff); return
    x-=pi.k1
    if x<0 or x>=pi.k2: return
    k=pi.id2pair2[x]; a=(k>>16)&0xffff; b=k&0xffff
    decode_id(pi,a,out,depth+1); decode_id(pi,b,out,depth+1)
def decode_ids(pi, ids):
    out=bytearray()
    for i in ids: decode_id(pi,i,out,0)
    return bytes(out)
def parse(s): return [int(x) for x in re.findall(r"<(\d+)>", s)]
pi = load_index_v7(Path(os.environ["INDEX_BIN"]))
s = sys.argv[1]
ids = parse(s)
raw = decode_ids(pi, ids)
print(raw.decode("utf-8", errors="replace"))
