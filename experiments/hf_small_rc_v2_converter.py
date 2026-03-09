import os, json, glob, struct, math
from pathlib import Path
import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open

repo_id = os.environ["HF_MODEL"]
workdir = Path(os.environ["WORKDIR"])
index_bin = Path(os.environ["INDEX_BIN"])
pair_k = int(os.environ["PAIR_K"])
pair_k1 = int(os.environ["PAIR_K1"])
D = int(os.environ["D"])
H = int(os.environ["H"])
L_tgt = int(os.environ["L"])
F_tgt = int(os.environ["F"])
TMAX = int(os.environ["TMAX"])
env_file = Path(os.environ["ENV_FILE"])
ckpt_file = Path(os.environ["CKPT_FILE"])
manifest_file = Path(os.environ["MANIFEST_FILE"])

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

def pool_1d(v, out_n):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    out = np.zeros((out_n,), dtype=np.float32)
    for i,(a,b) in enumerate(segment_bounds(v.shape[0], out_n)):
        out[i] = float(np.mean(v[a:b]))
    return out

def pool_axis_mean(arr, axis, out_n):
    arr = np.asarray(arr, dtype=np.float32)
    pieces=[]
    src_n = arr.shape[axis]
    for a,b in segment_bounds(src_n, out_n):
        sl=[slice(None)]*arr.ndim
        sl[axis]=slice(a,b)
        pieces.append(np.mean(arr[tuple(sl)], axis=axis, keepdims=True))
    return np.concatenate(pieces, axis=axis)

def pool_2d(m, out_shape):
    m = np.asarray(m, dtype=np.float32)
    if m.ndim != 2:
        m = m.reshape(m.shape[0], -1) if m.ndim > 1 else m.reshape(1,-1)
    out = pool_axis_mean(m, 0, out_shape[0])
    out = pool_axis_mean(out, 1, out_shape[1])
    return out.astype(np.float32, copy=False)

def transpose2d(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32).T)

def write_i32(f, arr, scale=1000.0):
    q = np.rint(np.asarray(arr, dtype=np.float32) * scale).astype(np.int32, copy=False).reshape(-1)
    f.write(q.tobytes(order="C"))

print(f"[*] Downloading HF metadata + safetensors for {repo_id} ...", flush=True)
model_path = Path(snapshot_download(repo_id, allow_patterns=["*.safetensors", "*.json", "config.json"]))

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
        for k in ks: key_to_file[k] = fp

def load_key(name):
    fp = key_to_file.get(name)
    if fp is None:
        return None
    with safe_open(fp, framework="np") as sf:
        try:
            return np.asarray(sf.get_tensor(name), dtype=np.float32)
        except Exception:
            return None

def load_any(candidates, required=False, label="tensor"):
    for name in candidates:
        arr = load_key(name)
        if arr is not None:
            print(f"[*] {label}: {name}", flush=True)
            return arr, name
    if required:
        prefix = "\n".join(sorted(all_keys[:120]))
        raise SystemExit(f"missing {label}; tried {candidates}\nfirst keys:\n{prefix}")
    return None, None

# robust aliases
embed, embed_key = load_any([
    "model.embed_tokens.weight",
    "tok_embeddings.weight",
    "transformer.wte.weight",
    "language_model.model.embed_tokens.weight",
    "backbone.embed_tokens.weight",
], required=True, label="embed_tokens")

final_norm, final_norm_key = load_any([
    "model.norm.weight",
    "norm.weight",
    "transformer.ln_f.weight",
    "language_model.model.norm.weight",
], required=False, label="final_norm")

lm_head, lm_head_key = load_any([
    "lm_head.weight",
    "output.weight",
    "embed_out.weight",
    "language_model.lm_head.weight",
], required=False, label="lm_head")
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
        return [x + s for x in p for s in ["self_attn.q_proj.weight", "attn.q_proj.weight", "self_attention.query_key_value.weight"]]
    if kind == "k":
        return [x + s for x in p for s in ["self_attn.k_proj.weight", "attn.k_proj.weight"]]
    if kind == "v":
        return [x + s for x in p for s in ["self_attn.v_proj.weight", "attn.v_proj.weight"]]
    if kind == "o":
        return [x + s for x in p for s in ["self_attn.o_proj.weight", "attn.o_proj.weight", "self_attention.dense.weight"]]
    if kind == "gate":
        return [x + s for x in p for s in ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.fc1.weight"]]
    if kind == "down":
        return [x + s for x in p for s in ["mlp.down_proj.weight", "mlp.down_proj.weight", "mlp.fc2.weight"]]
    return []

def choose_layer_map(src_L, dst_L):
    idx=[]
    for i in range(dst_L):
        x=(i+0.5)*src_L/dst_L - 0.5
        j=int(round(x))
        j=max(0,min(src_L-1,j))
        idx.append(j)
    for i in range(1,len(idx)):
        if idx[i] <= idx[i-1]:
            idx[i] = min(src_L-1, idx[i-1]+1)
    return idx

def mix_neighbors(src_L, centers, radius=1):
    return [sorted(set(max(0,min(src_L-1,c+d)) for d in range(-radius, radius+1))) for c in centers]

centers = choose_layer_map(L_src, L_tgt)
groups = mix_neighbors(L_src, centers, radius=1)

BASE_V = 256
V = BASE_V + pair_k
Vpad = ((V + 15) // 16) * 16
K2 = pair_k - pair_k1

with index_bin.open("rb") as idxf:
    index_data = idxf.read()
pow2 = struct.unpack("<I", index_data[16:20])[0]

manifest = {
    "repo_id": repo_id,
    "source_dims": {"D": D_src, "H": H_src, "L": L_src, "F": F_src, "T": T_src, "V": V_src},
    "target_dims": {"D": D, "H": H, "L": L_tgt, "F": F_tgt, "T": TMAX},
    "keys": {"embed": embed_key, "final_norm": final_norm_key, "lm_head": lm_head_key},
    "layer_groups": groups,
}

with env_file.open("w", encoding="utf-8") as f:
    f.write(f"export DMODEL={D}\nexport NHEAD={H}\nexport NLAY={L_tgt}\nexport FFN={F_tgt}\nexport TMAX={TMAX}\n")
    f.write(f"export CKPT_HASH={ckpt_file.stem.split('_')[-1]}\nexport DEFAULT_CKPT_FILE={ckpt_file}\n")

wte = normalize_rms(pool_2d(embed, (Vpad, D)), rms(embed))
wpe = np.zeros((TMAX, D), dtype=np.float32)
norm = pool_1d(final_norm if final_norm is not None else np.ones((D_src,), dtype=np.float32), D)
norm = normalize_rms(norm, rms(final_norm) if final_norm is not None else 1.0)
lmh = normalize_rms(pool_2d(transpose2d(lm_head), (D, Vpad)), rms(lm_head))

with ckpt_file.open("wb") as f:
    f.write(struct.pack("<II", 0x43484452, 11))
    f.write(struct.pack("<II", pair_k1, K2))
    f.write(struct.pack("<IIIII", D, H, L_tgt, F_tgt, TMAX))
    id2pair_len = pair_k1 * 2
    id2pair2_len = K2 * 4
    f.write(index_data[24 : 24 + id2pair_len])
    f.write(index_data[24 + id2pair_len : 24 + id2pair_len + id2pair2_len])
    f.write(struct.pack("<I", pow2))
    f.write(index_data[24 + id2pair_len + id2pair2_len:])

    write_i32(f, wte)
    write_i32(f, wpe)

    for i, group in enumerate(groups):
        def avg(kind):
            arrs=[]
            for src_l in group:
                cand = layer_key_candidates(src_l, kind)
                arr, key = load_any(cand, required=False, label=f"layer{src_l}_{kind}")
                if arr is not None:
                    arrs.append(arr)
            if not arrs:
                return None
            acc = np.zeros_like(arrs[0], dtype=np.float32)
            for a in arrs: acc += a
            return acc / float(len(arrs))

        r1m = avg("rms1")
        r2m = avg("rms2")
        qm = avg("q"); km = avg("k"); vm = avg("v"); om = avg("o")
        gm = avg("gate"); dm = avg("down")

        r1 = normalize_rms(pool_1d(r1m if r1m is not None else np.ones((D_src,), dtype=np.float32), Dhf_tgt), rms(r1m) if r1m is not None else 1.0)
        r2 = normalize_rms(pool_1d(r2m if r2m is not None else np.ones((D_src,), dtype=np.float32), Dhf_tgt), rms(r2m) if r2m is not None else 1.0)

        def proj2(a, shape):
            if a is None: return np.zeros(shape, dtype=np.float32)
            return normalize_rms(pool_2d(transpose2d(a), shape), rms(a))

        q = proj2(qm, (Dhf_tgt, Dhf_tgt))
        k = proj2(km, (Dhf_tgt, Dhf_tgt))
        v = proj2(vm, (Dhf_tgt, Dhf_tgt))
        o = proj2(om, (Dhf_tgt, Dhf_tgt))
        g = proj2(gm, (F_tgt, Dhf_tgt))
        d = proj2(dm, (Dhf_tgt, F_tgt))

        write_i32(f, r1); write_i32(f, q); write_i32(f, k); write_i32(f, v); write_i32(f, o); write_i32(f, r2); write_i32(f, g); write_i32(f, d)
        print(f"[*] Layer {i+1}/{L_tgt} compressed from {group}", flush=True)

    write_i32(f, norm)
    write_i32(f, lmh)

manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"[*] RC v2 native checkpoint done: {ckpt_file}", flush=True)
print(f"[*] Manifest: {manifest_file}", flush=True)
