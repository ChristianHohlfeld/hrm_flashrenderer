#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# run6_native_rc.sh
#
# Hardcore native RC bridge for YOUR stack.
#
# Zero Torch. Zero Accelerate. Zero llama.cpp.
# HF safetensors -> compressed native checkpoint -> run_llm_orig.sh
#
# Philosophy:
#   - preserve semantics harder than naive truncate/pad
#   - compress across the whole source model depth, not just first layers
#   - deterministic, reproducible, logged, manifest-driven
#
# What it does:
#   1) downloads HF model metadata + safetensors
#   2) loads weights with safetensors + numpy only
#   3) compresses into the hard small native core expected by run_llm_orig:
#        D=256, H=8, L=6, F=1024, TMAX=512  (defaults; D/H should stay fixed)
#   4) writes:
#        - native checkpoint
#        - manifest JSON
#        - env file
#   5) launches run_llm_orig.sh with the generated checkpoint
#
# Usage:
#   ./run6_native_rc.sh
#   HF_MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" ./run6_native_rc.sh
#   FORCE_NEW=1 ./run6_native_rc.sh
#   GPUS=1 ./run6_native_rc.sh
#
# IMPORTANT:
#   - This is a serious native compression bridge, not a full-fidelity 8B runtime.
#   - It targets your bounded native kernel on purpose.
# =============================================================================

need(){ command -v "$1" >/dev/null 2>&1 || { echo "FATAL: missing $1"; exit 1; }; }

need python3
need nvcc
need md5sum

WORKDIR="${WORKDIR:-$PWD}"
HF_MODEL="${HF_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
INDEX_BIN="${INDEX_BIN:-index_v7_k18192_k28192.bin}"
RUNNER="${RUNNER:-./run_llm_orig.sh}"

# Hard target core for run_llm_orig
DMODEL="${D:-256}"
NHEAD="${H:-8}"
NLAY="${L:-6}"
FFN="${F:-1024}"
TMAX="${TMAX:-512}"

PAIR_K="${PAIR_K:-16384}"
PAIR_K1="${PAIR_K1:-8192}"

FORCE_NEW="${FORCE_NEW:-1}"
GPUS="${GPUS:-2}"
MEASURE="${MEASURE:-1}"

if [[ ! -f "$RUNNER" ]]; then
  echo "FATAL: runner not found: $RUNNER"
  exit 1
fi

if [[ ! -f "$INDEX_BIN" ]]; then
  echo "FATAL: index not found: $INDEX_BIN"
  exit 1
fi

if [[ "$DMODEL" != "256" || "$NHEAD" != "8" ]]; then
  echo "FATAL: run_llm_orig hard kernel expects D=256 and H=8."
  exit 1
fi

if (( TMAX % 16 != 0 )); then
  echo "FATAL: TMAX must be multiple of 16"
  exit 1
fi

HASH_STR="llm_engine_K${PAIR_K}_D${DMODEL}_H${NHEAD}_L${NLAY}_F${FFN}_T${TMAX}"
CKPT_HASH="$(echo -n "$HASH_STR" | md5sum | head -c 8)"
CKPT_FILE="${WORKDIR}/ckpt_llm_engine_${CKPT_HASH}.bin"
ENV_FILE="${WORKDIR}/hf_env_small_rc.sh"
MANIFEST_FILE="${WORKDIR}/ckpt_llm_engine_${CKPT_HASH}.manifest.json"
CONVERTER="${WORKDIR}/hf_small_rc_converter.py"

if [[ "$FORCE_NEW" == "1" ]]; then
  rm -f "$CKPT_FILE" "$MANIFEST_FILE"
fi

cat > "$CONVERTER" <<'PY'
import os, sys, json, glob, struct, math, hashlib
import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open

repo_id       = os.environ["HF_MODEL"]
workdir       = os.environ["WORKDIR"]
index_bin     = os.environ["INDEX_BIN"]
pair_k        = int(os.environ["PAIR_K"])
pair_k1       = int(os.environ["PAIR_K1"])
D             = int(os.environ["DMODEL"])
H             = int(os.environ["NHEAD"])
L_tgt         = int(os.environ["NLAY"])
F_tgt         = int(os.environ["FFN"])
TMAX          = int(os.environ["TMAX"])
env_file      = os.environ["ENV_FILE"]
ckpt_file     = os.environ["CKPT_FILE"]
manifest_file = os.environ["MANIFEST_FILE"]

Dhf_tgt = D // 2
Dh_tgt  = Dhf_tgt // H

print(f"[*] Downloading HF metadata + safetensors for {repo_id} ...", flush=True)
model_path = snapshot_download(repo_id, allow_patterns=["*.safetensors", "config.json", "*.json"])

with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)

D_src = int(cfg.get("hidden_size", 4096))
H_src = int(cfg.get("num_attention_heads", 32))
L_src = int(cfg.get("num_hidden_layers", 32))
F_src = int(cfg.get("intermediate_size", 11008))
T_src = int(cfg.get("max_position_embeddings", 2048))
V_src = int(cfg.get("vocab_size", 32000))
H_kv  = int(cfg.get("num_key_value_heads", H_src))

manifest = {
    "repo_id": repo_id,
    "source_dims": {"D": D_src, "H": H_src, "L": L_src, "F": F_src, "T": T_src, "V": V_src, "H_kv": H_kv},
    "target_dims": {"D": D, "H": H, "L": L_tgt, "F": F_tgt, "T": TMAX},
    "compression": {},
}

print(f"[*] Source dims: D={D_src} H={H_src} L={L_src} F={F_src} T={T_src} V={V_src}", flush=True)
print(f"[*] Target dims: D={D} H={H} L={L_tgt} F={F_tgt} T={TMAX}", flush=True)

files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
if not files:
    raise SystemExit("no safetensors found")

key_to_file = {}
for fp in files:
    with safe_open(fp, framework="np") as sf:
        for k in sf.keys():
            key_to_file[k] = fp

def load_np(name):
    fp = key_to_file.get(name)
    if fp is None:
        return None
    with safe_open(fp, framework="np") as sf:
        try:
            arr = sf.get_tensor(name)
        except Exception:
            return None
    return np.asarray(arr, dtype=np.float32)

def rms(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))

def normalize_rms(x, target_rms):
    cur = rms(x)
    if cur <= 1e-12 or target_rms <= 1e-12:
        return np.asarray(x, dtype=np.float32)
    return np.asarray(x, dtype=np.float32) * (target_rms / cur)

def segment_bounds(src_n, dst_n):
    out = []
    for i in range(dst_n):
        a = int(math.floor(i * src_n / dst_n))
        b = int(math.floor((i + 1) * src_n / dst_n))
        if b <= a:
            b = min(src_n, a + 1)
        out.append((a, b))
    return out

def pool_1d(v, out_n):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    src_n = v.shape[0]
    out = np.zeros((out_n,), dtype=np.float32)
    for i, (a, b) in enumerate(segment_bounds(src_n, out_n)):
        out[i] = float(np.mean(v[a:b]))
    return out

def pool_axis_mean(arr, axis, out_n):
    arr = np.asarray(arr, dtype=np.float32)
    src_n = arr.shape[axis]
    pieces = []
    for a, b in segment_bounds(src_n, out_n):
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(a, b)
        pieces.append(np.mean(arr[tuple(sl)], axis=axis, keepdims=True))
    return np.concatenate(pieces, axis=axis)

def pool_2d(m, out_shape):
    m = np.asarray(m, dtype=np.float32)
    if m.ndim != 2:
        m = m.reshape(m.shape[0], -1) if m.ndim > 1 else m.reshape(1, -1)
    out = pool_axis_mean(m, 0, out_shape[0])
    out = pool_axis_mean(out, 1, out_shape[1])
    return out.astype(np.float32, copy=False)

def fit_vocab_rows(m, out_rows, out_cols):
    # Reduce/expand rows, then cols, with RMS preservation.
    pooled = pool_2d(m, (out_rows, out_cols))
    return normalize_rms(pooled, rms(m))

def choose_layer_map(src_L, dst_L):
    # Evenly cover the whole depth, centered per bucket.
    idx = []
    for i in range(dst_L):
        x = (i + 0.5) * src_L / dst_L - 0.5
        j = int(round(x))
        j = max(0, min(src_L - 1, j))
        idx.append(j)
    # Ensure strictly nondecreasing unique-ish map by nudging forward.
    for i in range(1, len(idx)):
        if idx[i] <= idx[i - 1]:
            idx[i] = min(src_L - 1, idx[i - 1] + 1)
    idx[-1] = min(src_L - 1, idx[-1])
    return idx

def avg_layers(names):
    arrs = [load_np(n) for n in names]
    arrs = [a for a in arrs if a is not None]
    if not arrs:
        return None
    base = np.zeros_like(arrs[0], dtype=np.float32)
    for a in arrs:
        base += np.asarray(a, dtype=np.float32)
    return base / float(len(arrs))

def mix_neighbors(layer_centers, radius=1):
    groups = []
    for c in layer_centers:
        g = [max(0, min(L_src - 1, c + d)) for d in range(-radius, radius + 1)]
        g = sorted(set(g))
        groups.append(g)
    return groups

def transpose2d(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32).T)

def write_i32(f, arr, scale=1000.0):
    a = np.asarray(arr, dtype=np.float32)
    q = np.rint(a * scale).astype(np.int32, copy=False).reshape(-1)
    f.write(q.tobytes(order="C"))

BASE_V = 256
V = BASE_V + pair_k
Vpad = ((V + 15) // 16) * 16
K2 = pair_k - pair_k1

with open(index_bin, "rb") as idxf:
    index_data = idxf.read()
pow2 = struct.unpack("<I", index_data[16:20])[0]

layer_centers = choose_layer_map(L_src, L_tgt)
layer_groups = mix_neighbors(layer_centers, radius=1)
manifest["compression"]["layer_centers"] = layer_centers
manifest["compression"]["layer_groups"] = layer_groups
manifest["compression"]["method"] = {
    "embeddings": "row/col mean-pool + RMS preserve",
    "attention": "neighbor-averaged layers + transpose + 2D mean-pool + RMS preserve",
    "mlp": "neighbor-averaged layers + transpose + 2D mean-pool + RMS preserve",
    "norms": "neighbor-averaged + 1D pool + RMS preserve",
    "lm_head": "transpose + 2D mean-pool + RMS preserve",
}

with open(env_file, "w", encoding="utf-8") as f:
    f.write(f"export DMODEL={D}\n")
    f.write(f"export NHEAD={H}\n")
    f.write(f"export NLAY={L_tgt}\n")
    f.write(f"export FFN={F_tgt}\n")
    f.write(f"export TMAX={TMAX}\n")
    f.write(f"export CKPT_HASH={os.path.basename(ckpt_file).split('_')[-1].split('.')[0]}\n")
    f.write(f"export DEFAULT_CKPT_FILE={ckpt_file}\n")

if os.path.exists(ckpt_file):
    print(f"[*] Checkpoint already exists: {ckpt_file}", flush=True)
    with open(manifest_file, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    sys.exit(0)

print(f"[*] Writing RC native checkpoint: {ckpt_file}", flush=True)

# ----- embeddings -----
wte_src = load_np("model.embed_tokens.weight")
if wte_src is None:
    raise SystemExit("missing model.embed_tokens.weight")
wte = fit_vocab_rows(wte_src, Vpad, D)

# ----- positional -----
wpe = np.zeros((TMAX, D), dtype=np.float32)

# ----- final norm / lm head -----
norm_src = load_np("model.norm.weight")
norm = pool_1d(norm_src if norm_src is not None else np.ones((D_src,), dtype=np.float32), D)
norm = normalize_rms(norm, rms(norm_src) if norm_src is not None else 1.0)

lm_head_src = load_np("lm_head.weight")
if lm_head_src is None:
    lm_head_src = wte_src  # fallback tie
lm_head = fit_vocab_rows(transpose2d(lm_head_src), D, Vpad)

with open(ckpt_file, "wb") as f:
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

    for li, group in enumerate(layer_groups):
        q_list, k_list, v_list, o_list = [], [], [], []
        rms1_list, rms2_list = [], []
        gate_list, down_list = [], []

        for src_l in group:
            q = load_np(f"model.layers.{src_l}.self_attn.q_proj.weight")
            k = load_np(f"model.layers.{src_l}.self_attn.k_proj.weight")
            v = load_np(f"model.layers.{src_l}.self_attn.v_proj.weight")
            o = load_np(f"model.layers.{src_l}.self_attn.o_proj.weight")
            if q is not None: q_list.append(q)
            if k is not None: k_list.append(k)
            if v is not None: v_list.append(v)
            if o is not None: o_list.append(o)

            r1 = load_np(f"model.layers.{src_l}.input_layernorm.weight")
            r2 = load_np(f"model.layers.{src_l}.post_attention_layernorm.weight")
            if r1 is not None: rms1_list.append(r1)
            if r2 is not None: rms2_list.append(r2)

            g = load_np(f"model.layers.{src_l}.mlp.gate_proj.weight")
            d = load_np(f"model.layers.{src_l}.mlp.down_proj.weight")
            if g is not None: gate_list.append(g)
            if d is not None: down_list.append(d)

        def mean_arr(lst):
            if not lst:
                return None
            acc = np.zeros_like(np.asarray(lst[0], dtype=np.float32))
            for a in lst:
                acc += np.asarray(a, dtype=np.float32)
            return acc / float(len(lst))

        q_m = mean_arr(q_list); k_m = mean_arr(k_list); v_m = mean_arr(v_list); o_m = mean_arr(o_list)
        r1_m = mean_arr(rms1_list); r2_m = mean_arr(rms2_list)
        g_m = mean_arr(gate_list); d_m = mean_arr(down_list)

        r1 = pool_1d(r1_m if r1_m is not None else np.ones((D_src,), dtype=np.float32), Dhf_tgt)
        r2 = pool_1d(r2_m if r2_m is not None else np.ones((D_src,), dtype=np.float32), Dhf_tgt)
        r1 = normalize_rms(r1, rms(r1_m) if r1_m is not None else 1.0)
        r2 = normalize_rms(r2, rms(r2_m) if r2_m is not None else 1.0)

        q_t = pool_2d(transpose2d(q_m) if q_m is not None else np.zeros((D_src, D_src), dtype=np.float32), (Dhf_tgt, Dhf_tgt))
        k_t = pool_2d(transpose2d(k_m) if k_m is not None else np.zeros((D_src, D_src), dtype=np.float32), (Dhf_tgt, Dhf_tgt))
        v_t = pool_2d(transpose2d(v_m) if v_m is not None else np.zeros((D_src, D_src), dtype=np.float32), (Dhf_tgt, Dhf_tgt))
        o_t = pool_2d(transpose2d(o_m) if o_m is not None else np.zeros((D_src, D_src), dtype=np.float32), (Dhf_tgt, Dhf_tgt))

        if q_m is not None: q_t = normalize_rms(q_t, rms(q_m))
        if k_m is not None: k_t = normalize_rms(k_t, rms(k_m))
        if v_m is not None: v_t = normalize_rms(v_t, rms(v_m))
        if o_m is not None: o_t = normalize_rms(o_t, rms(o_m))

        g_t = pool_2d(transpose2d(g_m) if g_m is not None else np.zeros((F_src, D_src), dtype=np.float32), (F_tgt, Dhf_tgt))
        d_t = pool_2d(transpose2d(d_m) if d_m is not None else np.zeros((D_src, F_src), dtype=np.float32), (Dhf_tgt, F_tgt))

        if g_m is not None: g_t = normalize_rms(g_t, rms(g_m))
        if d_m is not None: d_t = normalize_rms(d_t, rms(d_m))

        write_i32(f, r1)
        write_i32(f, q_t)
        write_i32(f, k_t)
        write_i32(f, v_t)
        write_i32(f, o_t)
        write_i32(f, r2)
        write_i32(f, g_t)
        write_i32(f, d_t)

        print(f"[*] Layer {li+1}/{L_tgt} compressed from source group {group}", flush=True)

    write_i32(f, norm)
    write_i32(f, lm_head)

manifest["output"] = {
    "ckpt_file": ckpt_file,
    "env_file": env_file,
    "manifest_file": manifest_file,
}

with open(manifest_file, "w", encoding="utf-8") as mf:
    json.dump(manifest, mf, indent=2)

print("[*] RC native checkpoint done.", flush=True)
print(f"[*] Manifest: {manifest_file}", flush=True)
PY

echo "[*] Running RC zero-torch converter..."
HF_MODEL="$HF_MODEL" WORKDIR="$WORKDIR" INDEX_BIN="$INDEX_BIN" \
PAIR_K="$PAIR_K" PAIR_K1="$PAIR_K1" DMODEL="$DMODEL" NHEAD="$NHEAD" NLAY="$NLAY" FFN="$FFN" TMAX="$TMAX" \
ENV_FILE="$ENV_FILE" CKPT_FILE="$CKPT_FILE" MANIFEST_FILE="$MANIFEST_FILE" \
python3 "$CONVERTER"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "FATAL: env file not generated"
  exit 1
fi

if [[ ! -f "$CKPT_FILE" ]]; then
  echo "FATAL: checkpoint not generated"
  exit 1
fi

source "$ENV_FILE"

echo "[*] Launching native CUDA runner..."
CMD=( "$RUNNER" --chat --gpus "$GPUS" --ckpt "$CKPT_FILE" )
if [[ "$MEASURE" == "1" ]]; then
  CMD+=( --measure )
fi
echo "[*] Manifest: $MANIFEST_FILE"
echo "[*] Command: ${CMD[*]}"
exec "${CMD[@]}"
