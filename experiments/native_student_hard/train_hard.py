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
V = BASE_V + PAIR_K
Vpad = ((V + 15) // 16) * 16
K2 = PAIR_K - PAIR_K1
Dhf = D // 2
EOS_ID = 1
SEP_ID = 2

def fnv1a(data: bytes) -> int:
    h = 2166136261
    for b in data:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h

def split_words(s: str):
    out=[]; cur=[]
    for ch in s:
        if ch.isalnum():
            cur.append(ch.lower())
        else:
            if cur:
                out.append("".join(cur)); cur=[]
    if cur: out.append("".join(cur))
    return out

def encode_pho(text: str):
    words = split_words(text)
    ids = []
    for w in words:
        a = w[:3].encode("utf-8", errors="ignore")
        b = w[-3:].encode("utf-8", errors="ignore")
        ids.append(BASE_V + (fnv1a(a) % PAIR_K))
        if len(ids) < 128:
            ids.append(BASE_V + (fnv1a(b) % PAIR_K))
    if not ids:
        ids = [BASE_V + (fnv1a(b"empty") % PAIR_K)]
    return ids[:TEACHER_LEN]

rows = []
with CALIB_JSONL.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        obj = json.loads(line)
        prompt = obj.get("prompt", "").strip()
        teacher_text = obj.get("teacher_text", "").strip()
        if not prompt or not teacher_text:
            continue
        p = encode_pho(prompt)
        t = encode_pho(teacher_text)
        seq = p + [SEP_ID] + t + [EOS_ID]
        rows.append({"prompt": prompt, "teacher_text": teacher_text, "prompt_ids": p, "target_ids": seq[:TEACHER_LEN+len(p)+2]})

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
        loss = 0.0
        used = target_ids[:TEACHER_LEN]
        prev = SEP_ID

        for tid in used:
            h, logits = self.step(h, prev)
            logits = logits - np.max(logits)
            p = np.exp(logits)
            p /= np.sum(p) + 1e-9
            loss += -math.log(float(p[tid]) + 1e-9)

            grad = p
            grad[tid] -= 1.0

            self.Wo -= lr * np.outer(h, grad)
            self.bo -= lr * grad

            # very cheap recurrent credit assignment approximation
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
            logits[:BASE_V] -= 0.35
            logits[EOS_ID] += 0.25
            if last is not None:
                logits[last] -= 0.25
            tid = int(np.argmax(logits))
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

def decode_quality(pred):
    if not pred:
        return -100.0
    non_special = sum(1 for x in pred if x >= BASE_V)
    eos = 1 if EOS_ID in pred else 0
    unique = len(set(pred))
    rep_penalty = len(pred) - unique
    return non_special * 1.5 + eos * 2.0 - rep_penalty * 0.5

def eval_student(stu, split_rows):
    losses = []; ovs = []; quals = []
    for row in split_rows:
        pred = stu.generate(row["prompt_ids"], max_len=48)
        ovs.append(overlap_score(pred, row["target_ids"]))
        quals.append(decode_quality(pred))
        h = stu.init_state(row["prompt_ids"])
        _, logits = stu.step(h, SEP_ID)
        tid = row["target_ids"][0]
        logits = logits - np.max(logits)
        p = np.exp(logits); p /= np.sum(p) + 1e-9
        losses.append(-math.log(float(p[tid]) + 1e-9))
    nll = float(np.mean(losses))
    ov = float(np.mean(ovs))
    q = float(np.mean(quals))
    return {"nll": nll, "overlap": ov, "quality": q, "score": ov * 120.0 + q * 3.0 - nll * 10.0}

def qint(a, scale=1000.0):
    return np.rint(np.asarray(a, dtype=np.float32) * scale).astype(np.int32, copy=False)

def build_native_ckpt(stu, ckpt_path: Path, manifest_path: Path, metrics: dict):
    with INDEX_BIN.open("rb") as idxf:
        index_data = idxf.read()
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
    h = min(Dhf, stu.hidden)
    proj_q[:h, :h] = np.eye(h, dtype=np.float32) * 0.6
    proj_k[:h, :h] = np.eye(h, dtype=np.float32) * 0.6
    proj_v[:h, :h] = np.eye(h, dtype=np.float32) * 0.6
    proj_o[:h, :h] = np.eye(h, dtype=np.float32) * 0.6

    gate = np.zeros((F, Dhf), dtype=np.float32)
    down = np.zeros((Dhf, F), dtype=np.float32)
    hh = min(Dhf, stu.hidden)
    ff = min(F, stu.hidden)
    gate[:ff, :hh] = stu.Wxh[:hh, :ff].T[:ff, :hh]
    down[:hh, :ff] = stu.Whh[:hh, :ff]

    norm = np.ones((D,), dtype=np.float32)
    lm_head = np.zeros((D, Vpad), dtype=np.float32)
    cols = min(D, stu.hidden)
    lm_head[:cols, :] = stu.Wo[:cols, :]
    # stronger EOS / separator calibration into head
    lm_head[0, EOS_ID] += 400.0
    lm_head[1, SEP_ID] += 200.0

    with ckpt_path.open("wb") as f:
        f.write(struct.pack("<II", 0x43484452, 11))
        f.write(struct.pack("<II", PAIR_K1, K2))
        f.write(struct.pack("<IIIII", D, H, L, F, TMAX))

        id2pair_len = PAIR_K1 * 2
        id2pair2_len = K2 * 4
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
        lr_ep = LR * (0.90 ** ep)
        for row in train_rows:
            losses.append(stu.train_example(row["prompt_ids"], row["target_ids"], lr_ep))
        tr = float(np.mean(losses))
        ev = eval_student(stu, val_rows)
        snapshot = {"epoch": ep + 1, "train_loss": tr, "val": ev}
        score = ev["score"] - tr * 5.0
        if best_score is None or score > best_score:
            best_score = score
            best_metrics = snapshot

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
