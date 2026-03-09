#!/usr/bin/env bash
set -euo pipefail

need(){ command -v "$1" >/dev/null 2>&1 || { echo "FATAL: missing $1"; exit 1; }; }

need python3
need nvcc
need md5sum

MODE="${1:---loop}"
WORKDIR="${WORKDIR:-$PWD}"
HF_MODEL="${HF_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
INDEX_BIN="${INDEX_BIN:-index_v7_k18192_k28192.bin}"
RUNNER="${RUNNER:-./run_llm_orig.sh}"
PAIR_K="${PAIR_K:-16384}"
PAIR_K1="${PAIR_K1:-8192}"
FORCE_NEW="${FORCE_NEW:-1}"
GPUS="${GPUS:-2}"
MEASURE="${MEASURE:-1}"

# hard kernel
export D=256
export H=8
export L=6
export F=1024
export TMAX=512

if [[ ! -f "$RUNNER" ]]; then echo "FATAL: runner not found: $RUNNER"; exit 1; fi
if [[ ! -f "$INDEX_BIN" ]]; then echo "FATAL: index not found: $INDEX_BIN"; exit 1; fi

HASH_BASE="llm_engine_K${PAIR_K}_D${D}_H${H}_L${L}_F${F}_T${TMAX}"
BASE_HASH="$(echo -n "$HASH_BASE" | md5sum | head -c 8)"
DISTILL_DIR="${WORKDIR}/native_last_stand_${BASE_HASH}"
mkdir -p "$DISTILL_DIR"

PYFILE="${DISTILL_DIR}/native_last_stand.py"
SELECTOR_JSON="${DISTILL_DIR}/selector_results.json"
FINAL_ENV="${DISTILL_DIR}/selected_env.sh"
FINAL_MANIFEST="${DISTILL_DIR}/selected_manifest.json"

if [[ "$FORCE_NEW" == "1" && "$MODE" != "--select-only" ]]; then
  rm -rf "${DISTILL_DIR:?}/"*
fi

cat > "$PYFILE" <<'PY'
import os, sys, json, glob, struct, math, hashlib, subprocess, re, statistics
from pathlib import Path
import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open

mode = os.environ["MODE"]
repo_id = os.environ["HF_MODEL"]
workdir = Path(os.environ["WORKDIR"])
distill_dir = Path(os.environ["DISTILL_DIR"])
index_bin = Path(os.environ["INDEX_BIN"])
runner = os.environ["RUNNER"]
pair_k = int(os.environ["PAIR_K"])
pair_k1 = int(os.environ["PAIR_K1"])
D = int(os.environ["D"]); H = int(os.environ["H"]); L_tgt = int(os.environ["L"]); F_tgt = int(os.environ["F"]); TMAX = int(os.environ["TMAX"])
gpus = int(os.environ["GPUS"])
measure = os.environ.get("MEASURE", "1") == "1"
selector_json = Path(os.environ["SELECTOR_JSON"])
final_env = Path(os.environ["FINAL_ENV"])
final_manifest = Path(os.environ["FINAL_MANIFEST"])
Dhf_tgt = D // 2

def rms(x):
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12)) if x.size else 0.0

def normalize_rms(x, target_rms):
    cur = rms(x)
    if cur <= 1e-12 or target_rms <= 1e-12:
        return np.asarray(x, dtype=np.float32)
    return np.asarray(x, dtype=np.float32) * (target_rms / cur)

def segment_bounds(src_n, dst_n):
    out=[]
    for i in range(dst_n):
        a=int(math.floor(i*src_n/dst_n))
        b=int(math.floor((i+1)*src_n/dst_n))
        if b <= a: b=min(src_n, a+1)
        out.append((a,b))
    return out

def pool_1d(v, out_n, mode="mean"):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    out = np.zeros((out_n,), dtype=np.float32)
    for i,(a,b) in enumerate(segment_bounds(v.shape[0], out_n)):
        seg = v[a:b]
        if mode == "absmax":
            j = int(np.argmax(np.abs(seg))); out[i] = float(seg[j])
        elif mode == "edge":
            out[i] = float(seg[0])
        else:
            out[i] = float(np.mean(seg))
    return out

def pool_axis(arr, axis, out_n, mode="mean"):
    arr = np.asarray(arr, dtype=np.float32)
    pieces=[]
    src_n = arr.shape[axis]
    for a,b in segment_bounds(src_n, out_n):
        sl=[slice(None)]*arr.ndim
        sl[axis]=slice(a,b)
        seg = arr[tuple(sl)]
        if mode == "absmax":
            idx = np.argmax(np.abs(seg), axis=axis)
            piece = np.take_along_axis(seg, np.expand_dims(idx, axis=axis), axis=axis)
        elif mode == "edge":
            sl2=[slice(None)]*arr.ndim; sl2[axis]=slice(a, a+1)
            piece = arr[tuple(sl2)]
        else:
            piece = np.mean(seg, axis=axis, keepdims=True)
        pieces.append(piece)
    return np.concatenate(pieces, axis=axis)

def pool_2d(m, out_shape, mode="mean"):
    m = np.asarray(m, dtype=np.float32)
    if m.ndim != 2:
        m = m.reshape(m.shape[0], -1) if m.ndim > 1 else m.reshape(1,-1)
    out = pool_axis(m, 0, out_shape[0], mode=mode)
    out = pool_axis(out, 1, out_shape[1], mode=mode)
    return out.astype(np.float32, copy=False)

def transpose2d(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32).T)

def write_i32(f, arr, scale=1000.0):
    q = np.rint(np.asarray(arr, dtype=np.float32) * scale).astype(np.int32, copy=False).reshape(-1)
    f.write(q.tobytes(order="C"))

def choose_even_layers(src_L, dst_L):
    idx = []
    for i in range(dst_L):
        x = (i + 0.5) * src_L / dst_L - 0.5
        j = int(round(x))
        j = max(0, min(src_L - 1, j))
        idx.append(j)
    for i in range(1, len(idx)):
        if idx[i] <= idx[i-1]:
            idx[i] = min(src_L - 1, idx[i-1] + 1)
    return idx

def groups_from_centers(src_L, centers, radius):
    return [sorted(set(max(0, min(src_L - 1, c + d)) for d in range(-radius, radius + 1))) for c in centers]

def run_probe(ckpt_path, prompt):
    cmd = [runner, "--chat", "--gpus", str(gpus), "--ckpt", str(ckpt_path)]
    if measure:
        cmd.append("--measure")
    try:
        proc = subprocess.run(
            cmd,
            input=(prompt + "\n/quit\n").encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=35,
            cwd=str(workdir),
        )
        out = proc.stdout.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf-8", errors="replace") + "\n[TIMEOUT]"
    return out

def extract_reply(log_text):
    lines = log_text.splitlines()
    seen = False
    out = []
    for ln in lines:
        if ln.startswith("> "):
            if seen and out:
                break
            seen = True
            continue
        if seen:
            if "[tok/s:" in ln:
                break
            if "commands:" in ln:
                continue
            if ln.strip():
                out.append(ln.rstrip())
    return "\n".join(out).strip()

def score_text(txt):
    if not txt:
        return -1000.0
    n = len(txt)
    printable = sum(ch.isprintable() for ch in txt)
    letters = sum(ch.isalpha() for ch in txt)
    spaces = txt.count(" ")
    weird = txt.count(" ") + txt.count("\x00")
    uniq = len(set(txt))
    score = 0.0
    score += min(n, 200) * 0.7
    score += printable / max(n,1) * 40.0
    score += letters / max(n,1) * 35.0
    score += min(spaces, 20) * 1.5
    score += min(uniq, 40) * 0.4
    score -= weird * 15.0
    if n < 4: score -= 250.0
    return score

print(f"[*] Downloading HF metadata + safetensors for {repo_id} ...", flush=True)
model_path = Path(snapshot_download(repo_id, allow_patterns=["*.safetensors","*.json","config.json"]))
cfg = json.loads((model_path / "config.json").read_text(encoding="utf-8"))

D_src = int(cfg.get("hidden_size", 4096))
H_src = int(cfg.get("num_attention_heads", 32))
L_src = int(cfg.get("num_hidden_layers", 32))
F_src = int(cfg.get("intermediate_size", 11008))
T_src = int(cfg.get("max_position_embeddings", 2048))
V_src = int(cfg.get("vocab_size", 32000))

print(f"[*] Source dims: D={D_src} H={H_src} L={L_src} F={F_src} T={T_src} V={V_src}", flush=True)
print(f"[*] Target dims: D={D} H={H} L={L_tgt} F={F_tgt} T={TMAX}", flush=True)

files = sorted(glob.glob(str(model_path / "*.safetensors")))
if not files:
    raise SystemExit("no safetensors found")

key_to_file = {}
all_keys = []
for fp in files:
    with safe_open(fp, framework="np") as sf:
        ks = list(sf.keys())
        all_keys.extend(ks)
        for k in ks:
            key_to_file[k] = fp

def load_key(name):
    fp = key_to_file.get(name)
    if fp is None:
        return None
    with safe_open(fp, framework="np") as sf:
        try:
            return np.asarray(sf.get_tensor(name), dtype=np.float32)
        except Exception:
            return None

def load_any(candidates, required=False):
    for name in candidates:
        arr = load_key(name)
        if arr is not None:
            return arr, name
    if required:
        raise SystemExit("missing required tensor; first keys:\n" + "\n".join(sorted(all_keys[:200])))
    return None, None

embed, embed_key = load_any([
    "model.embed_tokens.weight",
    "tok_embeddings.weight",
    "transformer.wte.weight",
    "language_model.model.embed_tokens.weight",
    "backbone.embed_tokens.weight",
], required=True)

final_norm, final_norm_key = load_any([
    "model.norm.weight",
    "norm.weight",
    "transformer.ln_f.weight",
    "language_model.model.norm.weight",
], required=False)

lm_head, lm_head_key = load_any([
    "lm_head.weight",
    "output.weight",
    "embed_out.weight",
    "language_model.lm_head.weight",
], required=False)
if lm_head is None:
    lm_head = embed
    lm_head_key = embed_key

def layer_key_candidates(layer, kind):
    p = [
        f"model.layers.{layer}.",
        f"layers.{layer}.",
        f"language_model.model.layers.{layer}.",
        f"transformer.h.{layer}.",
    ]
    if kind == "rms1":
        return [x + s for x in p for s in ["input_layernorm.weight", "ln_1.weight"]]
    if kind == "rms2":
        return [x + s for x in p for s in ["post_attention_layernorm.weight", "ln_2.weight"]]
    if kind == "q":
        return [x + s for x in p for s in ["self_attn.q_proj.weight", "attn.q_proj.weight"]]
    if kind == "k":
        return [x + s for x in p for s in ["self_attn.k_proj.weight", "attn.k_proj.weight"]]
    if kind == "v":
        return [x + s for x in p for s in ["self_attn.v_proj.weight", "attn.v_proj.weight"]]
    if kind == "o":
        return [x + s for x in p for s in ["self_attn.o_proj.weight", "attn.o_proj.weight", "self_attention.dense.weight"]]
    if kind == "gate":
        return [x + s for x in p for s in ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.fc1.weight"]]
    if kind == "down":
        return [x + s for x in p for s in ["mlp.down_proj.weight", "mlp.fc2.weight"]]
    return []

BASE_V = 256
V = BASE_V + pair_k
Vpad = ((V + 15) // 16) * 16
K2 = pair_k - pair_k1
with index_bin.open("rb") as idxf:
    index_data = idxf.read()
pow2 = struct.unpack("<I", index_data[16:20])[0]

strategies = [
    {"name": "mean_r1", "pool": "mean", "radius": 1, "scale": 1000.0},
    {"name": "mean_r2", "pool": "mean", "radius": 2, "scale": 900.0},
    {"name": "absmax_r1", "pool": "absmax", "radius": 1, "scale": 850.0},
    {"name": "edge_r1", "pool": "edge", "radius": 1, "scale": 1000.0},
]

centers = choose_even_layers(L_src, L_tgt)

def avg_kind(group, kind):
    arrs=[]
    used=[]
    for src_l in group:
        arr, key = load_any(layer_key_candidates(src_l, kind), required=False)
        if arr is not None:
            arrs.append(arr); used.append(key)
    if not arrs:
        return None, used
    acc = np.zeros_like(arrs[0], dtype=np.float32)
    for a in arrs:
        acc += np.asarray(a, dtype=np.float32)
    return acc / float(len(arrs)), used

def compress_matrix(a, out_shape, strat):
    if a is None:
        return np.zeros(out_shape, dtype=np.float32)
    return normalize_rms(pool_2d(a, out_shape, mode=strat["pool"]), rms(a))

def compress_vector(a, out_n, strat):
    if a is None:
        return np.ones((out_n,), dtype=np.float32)
    return normalize_rms(pool_1d(a, out_n, mode=strat["pool"]), rms(a))

def build_candidate(strat):
    groups = groups_from_centers(L_src, centers, strat["radius"])
    ckpt_hash = hashlib.md5(f"{repo_id}|{strat['name']}|D{D}H{H}L{L_tgt}F{F_tgt}T{TMAX}|K{pair_k}".encode()).hexdigest()[:8]
    ckpt = distill_dir / f"ckpt_{strat['name']}_{ckpt_hash}.bin"
    envf = distill_dir / f"env_{strat['name']}_{ckpt_hash}.sh"
    manifest = distill_dir / f"manifest_{strat['name']}_{ckpt_hash}.json"

    with envf.open("w", encoding="utf-8") as f:
        f.write(f"export DMODEL={D}\nexport NHEAD={H}\nexport NLAY={L_tgt}\nexport FFN={F_tgt}\nexport TMAX={TMAX}\n")
        f.write(f"export DEFAULT_CKPT_FILE={ckpt}\n")

    info = {
        "strategy": strat,
        "ckpt": str(ckpt),
        "env": str(envf),
        "manifest": str(manifest),
        "keys": {"embed": embed_key, "final_norm": final_norm_key, "lm_head": lm_head_key},
        "groups": groups,
        "layers": []
    }

    wte = compress_matrix(embed, (Vpad, D), strat)
    wpe = np.zeros((TMAX, D), dtype=np.float32)
    norm = compress_vector(final_norm if final_norm is not None else np.ones((D_src,), dtype=np.float32), D, strat)
    lmh = compress_matrix(transpose2d(lm_head), (D, Vpad), strat)

    with ckpt.open("wb") as f:
        f.write(struct.pack("<II", 0x43484452, 11))
        f.write(struct.pack("<II", pair_k1, K2))
        f.write(struct.pack("<IIIII", D, H, L_tgt, F_tgt, TMAX))
        id2pair_len = pair_k1 * 2
        id2pair2_len = K2 * 4
        f.write(index_data[24 : 24 + id2pair_len])
        f.write(index_data[24 + id2pair_len : 24 + id2pair_len + id2pair2_len])
        f.write(struct.pack("<I", pow2))
        f.write(index_data[24 + id2pair_len + id2pair2_len:])
        write_i32(f, wte, scale=strat["scale"])
        write_i32(f, wpe, scale=strat["scale"])

        for i, group in enumerate(groups):
            r1m, used_r1 = avg_kind(group, "rms1")
            r2m, used_r2 = avg_kind(group, "rms2")
            qm, used_q = avg_kind(group, "q")
            km, used_k = avg_kind(group, "k")
            vm, used_v = avg_kind(group, "v")
            om, used_o = avg_kind(group, "o")
            gm, used_g = avg_kind(group, "gate")
            dm, used_d = avg_kind(group, "down")

            r1 = compress_vector(r1m if r1m is not None else np.ones((D_src,), dtype=np.float32), Dhf_tgt, strat)
            r2 = compress_vector(r2m if r2m is not None else np.ones((D_src,), dtype=np.float32), Dhf_tgt, strat)
            q = compress_matrix(transpose2d(qm) if qm is not None else np.zeros((D_src, D_src), dtype=np.float32), (Dhf_tgt, Dhf_tgt), strat)
            k = compress_matrix(transpose2d(km) if km is not None else np.zeros((D_src, D_src), dtype=np.float32), (Dhf_tgt, Dhf_tgt), strat)
            v = compress_matrix(transpose2d(vm) if vm is not None else np.zeros((D_src, D_src), dtype=np.float32), (Dhf_tgt, Dhf_tgt), strat)
            o = compress_matrix(transpose2d(om) if om is not None else np.zeros((D_src, D_src), dtype=np.float32), (Dhf_tgt, Dhf_tgt), strat)
            g = compress_matrix(transpose2d(gm) if gm is not None else np.zeros((F_src, D_src), dtype=np.float32), (F_tgt, Dhf_tgt), strat)
            d = compress_matrix(transpose2d(dm) if dm is not None else np.zeros((D_src, F_src), dtype=np.float32), (Dhf_tgt, F_tgt), strat)

            write_i32(f, r1, scale=strat["scale"]); write_i32(f, q, scale=strat["scale"]); write_i32(f, k, scale=strat["scale"])
            write_i32(f, v, scale=strat["scale"]); write_i32(f, o, scale=strat["scale"]); write_i32(f, r2, scale=strat["scale"])
            write_i32(f, g, scale=strat["scale"]); write_i32(f, d, scale=strat["scale"])

            info["layers"].append({"i": i, "group": group, "used": {"r1": used_r1, "r2": used_r2, "q": used_q, "k": used_k, "v": used_v, "o": used_o, "g": used_g, "d": used_d}})

        write_i32(f, norm, scale=strat["scale"])
        write_i32(f, lmh, scale=strat["scale"])

    manifest.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info

probe_prompts = ["hi", "how are you", "explain yourself in one sentence"]

def maybe_build():
    infos = []
    for strat in strategies:
        print(f"[*] Building candidate: {strat['name']}", flush=True)
        infos.append(build_candidate(strat))
    return infos

def score_candidates(infos):
    for info in infos:
        logs = []
        scores = []
        for p in probe_prompts:
            raw = run_probe(info["ckpt"], p)
            reply = extract_reply(raw)
            s = score_text(reply)
            logs.append({"prompt": p, "reply": reply, "score": s})
            scores.append(s)
            print(f"    prompt={p!r} score={s:.2f} reply={reply[:120]!r}", flush=True)
        info["probe"] = {
            "total_score": float(sum(scores)),
            "mean_score": float(statistics.mean(scores)) if scores else -1e9,
            "logs": logs,
        }
    infos = sorted(infos, key=lambda x: x["probe"]["total_score"], reverse=True)
    selector_json.write_text(json.dumps(infos, indent=2), encoding="utf-8")
    best = infos[0]
    final_env.write_text(
        f"export DMODEL={D}\nexport NHEAD={H}\nexport NLAY={L_tgt}\nexport FFN={F_tgt}\nexport TMAX={TMAX}\nexport DEFAULT_CKPT_FILE={best['ckpt']}\n",
        encoding="utf-8"
    )
    final_manifest.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"[*] Selected candidate: {Path(best['ckpt']).name} score={best['probe']['total_score']:.2f}", flush=True)
    return infos

def load_existing_infos():
    manifests = sorted(distill_dir.glob("manifest_*.json"))
    infos = []
    for m in manifests:
        try:
            infos.append(json.loads(m.read_text(encoding="utf-8")))
        except Exception:
            pass
    if not infos:
        raise SystemExit("no existing manifests/candidates found")
    return infos

if mode == "--build-only":
    maybe_build()
elif mode == "--select-only":
    infos = load_existing_infos()
    score_candidates(infos)
else:
    infos = maybe_build()
    score_candidates(infos)
PY

export MODE HF_MODEL WORKDIR DISTILL_DIR INDEX_BIN RUNNER PAIR_K PAIR_K1 GPUS MEASURE D H L F TMAX SELECTOR_JSON FINAL_ENV FINAL_MANIFEST
echo "[*] Running native last-stand builder..."
python3 "$PYFILE"

if [[ ! -f "$SELECTOR_JSON" ]]; then
  echo "FATAL: selector results missing"
  exit 1
fi

pick_and_chat() {
  python3 - <<'PY'
import json, os, subprocess, sys
from pathlib import Path
selector = Path(os.environ["SELECTOR_JSON"])
runner = os.environ["RUNNER"]
gpus = os.environ["GPUS"]
measure = os.environ.get("MEASURE", "1") == "1"
results = json.loads(selector.read_text(encoding="utf-8"))
print("\nCandidates:")
for i, item in enumerate(results):
    print(f"  [{i}] {Path(item['ckpt']).name}  score={item['probe']['total_score']:.2f}  strat={item['strategy']['name']}")
sel = input("\nPick candidate index (blank = best, q = quit): ").strip()
if sel.lower() == "q":
    sys.exit(0)
idx = 0 if not sel else int(sel)
item = results[idx]
cmd = [runner, "--chat", "--gpus", str(gpus), "--ckpt", item["ckpt"]]
if measure:
    cmd.append("--measure")
print("[*] Launching:", " ".join(cmd))
subprocess.run(cmd)
PY
}

case "$MODE" in
  --build-only)
    echo "[*] Build-only complete: $DISTILL_DIR"
    ;;
  --select-only)
    echo "[*] Select-only complete: $SELECTOR_JSON"
    ;;
  --best)
    BEST_CKPT="$(python3 - <<'PY'
import json, os
from pathlib import Path
j = json.loads(Path(os.environ["FINAL_MANIFEST"]).read_text(encoding="utf-8"))
print(j["ckpt"])
PY
)"
    CMD=( "$RUNNER" --chat --gpus "$GPUS" --ckpt "$BEST_CKPT" )
    if [[ "$MEASURE" == "1" ]]; then CMD+=( --measure ); fi
    echo "[*] Launching best candidate..."
    exec "${CMD[@]}"
    ;;
  --pick|--loop)
    pick_and_chat
    ;;
  *)
    echo "unknown mode: $MODE"
    exit 1
    ;;
esac
