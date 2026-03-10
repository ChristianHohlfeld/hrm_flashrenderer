import os, json, math, random, struct
from pathlib import Path
import numpy as np

WORKDIR = Path(os.environ["WORKDIR"])
CALIB_JSONL = Path(os.environ["CALIB_JSONL"])
INDEX_BIN = Path(os.environ["INDEX_BIN"])
OUT_DIR = Path(os.environ["OUT_DIR"])
D = int(os.environ["D"]); H = int(os.environ["H"]); L = int(os.environ["L"]); F = int(os.environ["F"]); TMAX = int(os.environ["TMAX"])
PAIR_K = int(os.environ["PAIR_K"]); PAIR_K1 = int(os.environ["PAIR_K1"])
EPOCHS = int(os.environ["EPOCHS"]); LR = float(os.environ["LR"]); HOLDOUT = float(os.environ["HOLDOUT_FRACTION"]); CANDIDATES = int(os.environ["CANDIDATES"]); HIDDEN = int(os.environ["HIDDEN"]); TEACHER_LEN = int(os.environ["TEACHER_LEN"]); SEED = int(os.environ["SEED"])

random.seed(SEED)
np.random.seed(SEED)

BASE_V = 256
EOS_ID = ord('\n')
SEP_ID = ord(':')

class PairIndex:
    def __init__(self):
        self.k1 = 0; self.k2 = 0; self.pow2 = 0
        self.id2pair = []; self.id2pair2 = []
        self.hkeys = []; self.hvals = []
        self.pair2id = {}

def load_index_v7(path: Path) -> PairIndex:
    b = path.read_bytes()
    magic, ver, k1, k2, pow2, res = struct.unpack_from("<6I", b, 0)
    if magic != 0x37584449:
        raise SystemExit(f"bad index magic: 0x{magic:08x}")
    off = 24
    mv = memoryview(b)
    pi = PairIndex()
    pi.k1 = k1; pi.k2 = k2; pi.pow2 = pow2
    id2pair_raw = mv[off : off + k1 * 2]
    pi.id2pair = [struct.unpack_from("<H", id2pair_raw, i * 2)[0] for i in range(k1)]
    off += k1 * 2
    for i, p in enumerate(pi.id2pair):
        pi.pair2id[p] = BASE_V + i
    id2pair2_raw = mv[off : off + k2 * 4]
    pi.id2pair2 = [struct.unpack_from("<I", id2pair2_raw, i * 4)[0] for i in range(k2)]
    off += k2 * 4
    table_size = 1 << pow2
    hkeys_raw = mv[off : off + table_size * 4]
    pi.hkeys = [struct.unpack_from("<I", hkeys_raw, i * 4)[0] for i in range(table_size)]
    off += table_size * 4
    hvals_raw = mv[off : off + table_size * 2]
    pi.hvals = [struct.unpack_from("<H", hvals_raw, i * 2)[0] for i in range(table_size)]
    return pi

pi = load_index_v7(INDEX_BIN)
V = BASE_V + pi.k1 + pi.k2
Vpad = ((V + 15) // 16) * 16
Dhf = D // 2
K2 = pi.k2

def stage2_lookup(pi: PairIndex, a: int, b: int):
    key = ((a & 0xFFFF) << 16) | (b & 0xFFFF)
    mask = (1 << pi.pow2) - 1
    i = ((key * 2654435761) & 0xFFFFFFFF) & mask
    start = i
    while True:
        hk = pi.hkeys[i]
        if hk == key:
            hv = pi.hvals[i]
            if hv >= pi.k2:
                return None
            out = BASE_V + pi.k1 + hv
            if out >= Vpad:
                return None
            return out
        if hk == 0:
            return None
        i = (i + 1) & mask
        if i == start:
            return None

def next_stage1_id(pi: PairIndex, b: bytes, i: int):
    n = len(b)
    if i + 1 < n:
        pair = b[i] | (b[i+1] << 8)
        x = pi.pair2id.get(pair)
        if x is not None and x < Vpad:
            return x, i + 2
    return b[i], i + 1

def encode_ids(pi: PairIndex, b: bytes):
    ids = []
    i = 0
    while i < len(b):
        x, i = next_stage1_id(pi, b, i)
        ids.append(x)

    changed = True
    while changed and len(ids) >= 2:
        changed = False
        out = []
        j = 0
        while j < len(ids):
            if j + 1 < len(ids):
                merged = stage2_lookup(pi, ids[j], ids[j+1])
                if merged is not None:
                    out.append(merged)
                    j += 2
                    changed = True
                    continue
            out.append(ids[j]); j += 1
        ids = out

    ids = [x for x in ids if 0 <= x < Vpad]
    return ids

def decode_id(pi: PairIndex, idx: int, out: bytearray, depth=0):
    if depth > 64:
        return
    if idx < BASE_V:
        out.append(idx & 0xFF); return
    x = idx - BASE_V
    if x < pi.k1:
        p = pi.id2pair[x]
        out.append(p & 0xFF); out.append((p >> 8) & 0xFF); return
    x -= pi.k1
    if x < 0 or x >= pi.k2:
        return
    k = pi.id2pair2[x]
    a = (k >> 16) & 0xFFFF
    b = k & 0xFFFF
    decode_id(pi, a, out, depth + 1)
    decode_id(pi, b, out, depth + 1)

def decode_ids(pi: PairIndex, ids):
    out = bytearray()
    for i in ids:
        if 0 <= i < Vpad:
            decode_id(pi, i, out, 0)
    return bytes(out)

rows = []
with CALIB_JSONL.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        prompt = obj.get("prompt", "").strip()
        teacher_text = obj.get("teacher_text", "").strip()
        if not prompt or not teacher_text:
            continue
        p = encode_ids(pi, prompt.encode("utf-8", errors="replace"))[:64]
        t = encode_ids(pi, teacher_text.encode("utf-8", errors="replace"))[:TEACHER_LEN]
        if not p or not t:
            continue
        seq = p + [SEP_ID] + t + [EOS_ID]
        seq = [x for x in seq if 0 <= x < Vpad]
        if len(seq) < 3:
            continue
        rows.append({"prompt": prompt, "teacher_text": teacher_text, "prompt_ids": p, "target_ids": seq})

if len(rows) < 100:
    raise SystemExit(f"Need at least 100 rows, got {len(rows)}")

random.shuffle(rows)
cut = max(1, int(len(rows) * (1.0 - HOLDOUT)))
train_rows = rows[:cut]
val_rows = rows[cut:]

class Student:
    def __init__(self, seed: int, hidden: int):
        rs = np.random.RandomState(seed)
        self.hidden = hidden
        self.E = (rs.randn(Vpad, hidden).astype(np.float32) * 0.02)
        self.Wxh = (rs.randn(hidden, hidden).astype(np.float32) * 0.04)
        self.Whh = (rs.randn(hidden, hidden).astype(np.float32) * 0.04)
        self.bh = np.zeros((hidden,), dtype=np.float32)
        self.Wo = (rs.randn(hidden, Vpad).astype(np.float32) * 0.02)
        self.bo = np.zeros((Vpad,), dtype=np.float32)

    def init_state(self, prompt_ids):
        x = np.mean(self.E[np.asarray(prompt_ids, dtype=np.int32)], axis=0)
        return np.tanh(x @ self.Wxh + self.bh)

    def step(self, h, tok_id):
        x = self.E[tok_id]
        h2 = np.tanh(x @ self.Wxh + h @ self.Whh + self.bh)
        logits = h2 @ self.Wo + self.bo
        return h2, logits

    def train_example(self, prompt_ids, target_ids, lr):
        h = self.init_state(prompt_ids)
        used = target_ids[:TEACHER_LEN]
        prev = SEP_ID
        loss = 0.0
        for tid in used:
            h, logits = self.step(h, prev)
            logits = logits - np.max(logits)
            p = np.exp(logits); p /= np.sum(p) + 1e-9
            loss += -math.log(float(p[tid]) + 1e-9)

            grad = p
            grad[tid] -= 1.0
            self.Wo -= lr * np.outer(h, grad)
            self.bo -= lr * grad

            dh = self.Wo @ grad
            h_raw = np.clip(h, -0.999, 0.999)
            dpre = dh * (1.0 - h_raw * h_raw)
            x = self.E[prev]

            self.Wxh -= lr * np.outer(x, dpre)
            self.Whh -= lr * np.outer(h, dpre)
            self.bh  -= lr * dpre
            self.E[prev] -= lr * (self.Wxh @ dpre)
            prev = tid

        return loss / max(1, len(used))

    def generate(self, prompt_ids, max_len=48):
        h = self.init_state(prompt_ids)
        prev = SEP_ID
        out = []
        repeat = 0
        last = None
        for _ in range(max_len):
            h, logits = self.step(h, prev)
            logits[:BASE_V] -= 0.10
            logits[EOS_ID] += 0.10
            if last is not None:
                logits[last] -= 0.15
            tid = int(np.argmax(logits))
            if tid < 0 or tid >= Vpad:
                break
            out.append(tid)
            if tid == EOS_ID:
                break
            if tid == last:
                repeat += 1
                if repeat >= 8:
                    break
            else:
                repeat = 0
            last = tid
            prev = tid
        return out

def overlap_score(pred, gold):
    ps = set(pred); gs = set(gold)
    return len(ps & gs) / max(1, len(gs))

def decode_quality(pi, pred):
    if not pred:
        return -100.0
    text = decode_ids(pi, pred).decode("utf-8", errors="replace")
    printable = sum(ch.isprintable() for ch in text)
    letters = sum(ch.isalpha() for ch in text)
    weird = text.count(" ")
    return printable + letters * 2 - weird * 5

def eval_student(stu, split_rows):
    losses = []; ovs = []; quals = []
    for row in split_rows:
        pred = stu.generate(row["prompt_ids"], max_len=48)
        ovs.append(overlap_score(pred, row["target_ids"]))
        quals.append(decode_quality(pi, pred))
        h = stu.init_state(row["prompt_ids"])
        _, logits = stu.step(h, SEP_ID)
        tid = row["target_ids"][0]
        logits = logits - np.max(logits)
        p = np.exp(logits); p /= np.sum(p) + 1e-9
        losses.append(-math.log(float(p[tid]) + 1e-9))
    nll = float(np.mean(losses))
    ov = float(np.mean(ovs))
    q = float(np.mean(quals))
    return {"nll": nll, "overlap": ov, "quality": q, "score": ov * 100.0 + q * 1.5 - nll * 10.0}

def qint(a, scale=1000.0):
    return np.rint(np.asarray(a, dtype=np.float32) * scale).astype(np.int32, copy=False)

def build_native_ckpt(stu, ckpt_path: Path, manifest_path: Path, metrics: dict):
    index_data = INDEX_BIN.read_bytes()
    pow2 = struct.unpack("<I", index_data[16:20])[0]

    wte = np.zeros((Vpad, D), dtype=np.float32)
    cols = min(D, stu.hidden)
    wte[:, :cols] = stu.E[:, :cols]

    wpe = np.zeros((TMAX, D), dtype=np.float32)
    r1 = np.ones((Dhf,), dtype=np.float32)
    r2 = np.ones((Dhf,), dtype=np.float32)

    proj_q = np.zeros((Dhf, Dhf), dtype=np.float32)
    proj_k = np.zeros((Dhf, Dhf), dtype=np.float32)
    proj_v = np.zeros((Dhf, Dhf), dtype=np.float32)
    proj_o = np.zeros((Dhf, Dhf), dtype=np.float32)
    hh = min(Dhf, stu.hidden)
    proj_q[:hh, :hh] = np.eye(hh, dtype=np.float32) * 0.6
    proj_k[:hh, :hh] = np.eye(hh, dtype=np.float32) * 0.6
    proj_v[:hh, :hh] = np.eye(hh, dtype=np.float32) * 0.6
    proj_o[:hh, :hh] = np.eye(hh, dtype=np.float32) * 0.6

    gate = np.zeros((F, Dhf), dtype=np.float32)
    down = np.zeros((Dhf, F), dtype=np.float32)
    ff = min(F, stu.hidden)
    gate[:ff, :hh] = stu.Wxh[:hh, :ff].T[:ff, :hh]
    down[:hh, :ff] = stu.Whh[:hh, :ff]

    norm = np.ones((D,), dtype=np.float32)
    lm_head = np.zeros((D, Vpad), dtype=np.float32)
    cols = min(D, stu.hidden)
    lm_head[:cols, :] = stu.Wo[:cols, :]
    lm_head[0, EOS_ID] += 200.0
    lm_head[1, SEP_ID] += 100.0

    with ckpt_path.open("wb") as f:
        f.write(struct.pack("<II", 0x43484452, 11))
        f.write(struct.pack("<II", pi.k1, pi.k2))
        f.write(struct.pack("<IIIII", D, H, L, F, TMAX))

        id2pair_len = pi.k1 * 2
        id2pair2_len = pi.k2 * 4
        f.write(index_data[24 : 24 + id2pair_len])
        f.write(index_data[24 + id2pair_len : 24 + id2pair_len + id2pair2_len])
        f.write(struct.pack("<I", pow2))
        f.write(index_data[24 + id2pair_len + id2pair2_len:])

        f.write(qint(wte).reshape(-1).tobytes())
        f.write(qint(wpe).reshape(-1).tobytes())
        for _ in range(L):
            f.write(qint(r1).reshape(-1).tobytes())
            f.write(qint(proj_q).reshape(-1).tobytes())
            f.write(qint(proj_k).reshape(-1).tobytes())
            f.write(qint(proj_v).reshape(-1).tobytes())
            f.write(qint(proj_o).reshape(-1).tobytes())
            f.write(qint(r2).reshape(-1).tobytes())
            f.write(qint(gate).reshape(-1).tobytes())
            f.write(qint(down).reshape(-1).tobytes())
        f.write(qint(norm).reshape(-1).tobytes())
        f.write(qint(lm_head).reshape(-1).tobytes())

    manifest = {
        "metrics": metrics,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "dims": {"D": D, "H": H, "L": L, "F": F, "TMAX": TMAX, "Vpad": Vpad},
        "ckpt": str(ckpt_path),
        "hidden": HIDDEN,
        "tokenizer": "true_pairindex_v7_fixed"
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

OUT_DIR.mkdir(parents=True, exist_ok=True)
candidates = []
for ci in range(CANDIDATES):
    seed = SEED + ci * 97
    stu = Student(seed=seed, hidden=HIDDEN)
    best_score = None
    best_metrics = None
    for ep in range(EPOCHS):
        random.shuffle(train_rows)
        losses = []
        lr_ep = LR * (0.92 ** ep)
        for row in train_rows:
            losses.append(stu.train_example(row["prompt_ids"], row["target_ids"], lr_ep))
        tr = float(np.mean(losses))
        ev = eval_student(stu, val_rows)
        snap = {"epoch": ep + 1, "train_loss": tr, "val": ev}
        score = ev["score"] - tr * 5.0
        if best_score is None or score > best_score:
            best_score = score
            best_metrics = snap

    ckpt = OUT_DIR / f"student_candidate_{ci}.bin"
    manifest = OUT_DIR / f"student_candidate_{ci}.manifest.json"
    build_native_ckpt(stu, ckpt, manifest, best_metrics)
    candidates.append({"candidate": ci, "seed": seed, "score": best_score, "metrics": best_metrics, "ckpt": str(ckpt), "manifest": str(manifest)})

candidates.sort(key=lambda x: x["score"], reverse=True)
best = candidates[0]
selector = {"best": best, "all": candidates}
(OUT_DIR / "selector_results.json").write_text(json.dumps(selector, indent=2), encoding="utf-8")
(OUT_DIR / "selected_manifest.json").write_text(Path(best["manifest"]).read_text(encoding="utf-8"), encoding="utf-8")
(OUT_DIR / "selected_env.sh").write_text(
    f"export DMODEL={D}\nexport NHEAD={H}\nexport NLAY={L}\nexport FFN={F}\nexport TMAX={TMAX}\nexport DEFAULT_CKPT_FILE={best['ckpt']}\n",
    encoding="utf-8"
)

print(f"[*] Trained {len(candidates)} candidates")
print(f"[*] Best checkpoint: {best['ckpt']}")
print(f"[*] Best score: {best['score']:.4f}")
print(f"[*] Selector: {OUT_DIR / 'selector_results.json'}")
