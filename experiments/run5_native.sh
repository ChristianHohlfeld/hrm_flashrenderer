#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# run5_native.sh
#
# Zero-Torch / zero-Accelerate HF -> native small-core bridge for YOUR stack.
#
# What it does:
#   1) Downloads HF safetensors + config.json
#   2) Converts them WITHOUT torch into a bounded native checkpoint
#      targeting the hard run_llm_orig kernel:
#         D=256, H=8, L=6, F=1024, TMAX=512 (defaults, overridable)
#   3) Sources the generated hf_env_small.sh
#   4) Runs your native CUDA stack through run_llm_orig.sh
#
# Why this path:
#   - run_llm_orig is hard-asserted to D=256/H=8/Dh=16
#   - run3 HF path maps large model dims directly and blows up VRAM/runtime
#   - this script intentionally COMPRESSES/TRUNCATES/PADS into the small core
#
# IMPORTANT:
#   - This is a compression/bridge path, not exact full-fidelity DeepSeek.
#   - It is designed for "low VRAM, actually runs" on your stack.
#
# Usage:
#   ./run5_native.sh
#   HF_MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" ./run5_native.sh
#   D=256 H=8 L=6 F=1024 TMAX=512 ./run5_native.sh
#   FORCE_NEW=1 ./run5_native.sh
# =============================================================================

need(){ command -v "$1" >/dev/null 2>&1 || { echo "FATAL: missing $1"; exit 1; }; }

need python3
need nvcc
need md5sum

WORKDIR="${WORKDIR:-$PWD}"
HF_MODEL="${HF_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
INDEX_BIN="${INDEX_BIN:-index_v7_k18192_k28192.bin}"
RUNNER="${RUNNER:-./run_llm_orig.sh}"
DATA_FILE="${DATA_FILE:-tinyshakespeare.txt}"

# Hard small-core defaults for run_llm_orig
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

if (( TMAX % 16 != 0 )); then
  echo "FATAL: TMAX must be multiple of 16"
  exit 1
fi

HASH_STR="llm_engine_K${PAIR_K}_D${DMODEL}_H${NHEAD}_L${NLAY}_F${FFN}_T${TMAX}"
CKPT_HASH="$(echo -n "$HASH_STR" | md5sum | head -c 8)"
CKPT_FILE="${WORKDIR}/ckpt_llm_engine_${CKPT_HASH}.bin"
ENV_FILE="${WORKDIR}/hf_env_small.sh"
CONVERTER="${WORKDIR}/hf_small_converter.py"

if [[ "$FORCE_NEW" == "1" ]]; then
  rm -f "$CKPT_FILE"
fi

cat > "$CONVERTER" <<'PY'
import os, sys, json, glob, struct, hashlib, math
import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open

repo_id      = os.environ["HF_MODEL"]
workdir      = os.environ["WORKDIR"]
index_bin    = os.environ["INDEX_BIN"]
pair_k       = int(os.environ["PAIR_K"])
pair_k1      = int(os.environ["PAIR_K1"])
D            = int(os.environ["DMODEL"])
H            = int(os.environ["NHEAD"])
L            = int(os.environ["NLAY"])
F            = int(os.environ["FFN"])
TMAX         = int(os.environ["TMAX"])
env_file     = os.environ["ENV_FILE"]
ckpt_file    = os.environ["CKPT_FILE"]

if D != 256 or H != 8:
    print("WARNING: run_llm_orig hard-asserts D=256,H=8; other values will not compile cleanly", flush=True)

print(f"[*] Downloading HF metadata + safetensors for {repo_id} ...", flush=True)
model_path = snapshot_download(repo_id, allow_patterns=["*.safetensors", "config.json", "*.json"])

with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)

D_orig = int(cfg.get("hidden_size", 4096))
H_orig = int(cfg.get("num_attention_heads", 32))
L_orig = int(cfg.get("num_hidden_layers", 32))
F_orig = int(cfg.get("intermediate_size", 11008))
T_orig = int(cfg.get("max_position_embeddings", 2048))
V_orig = int(cfg.get("vocab_size", 32000))
H_kv   = int(cfg.get("num_key_value_heads", H_orig))
Dhf_small = D // 2
Dh_small  = Dhf_small // H

print(f"[*] Source dims: D={D_orig} H={H_orig} L={L_orig} F={F_orig} T={T_orig} V={V_orig}", flush=True)
print(f"[*] Target dims: D={D} H={H} L={L} F={F} T={TMAX}", flush=True)

# Load tensor index
files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
if not files:
    raise SystemExit("no safetensors found")

def list_keys(sf):
    try:
        return list(sf.keys())
    except Exception:
        return []

key_to_file = {}
for fp in files:
    with safe_open(fp, framework="np") as sf:
        for k in list_keys(sf):
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

def zeros(shape):
    return np.zeros(shape, dtype=np.float32)

def fit_1d(arr, out_len):
    out = np.zeros((out_len,), dtype=np.float32)
    if arr is None:
        return out
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    n = min(out_len, arr.shape[0])
    out[:n] = arr[:n]
    return out

def fit_2d(arr, out_shape):
    out = np.zeros(out_shape, dtype=np.float32)
    if arr is None:
        return out
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(1, -1)
    r = min(out_shape[0], arr.shape[0])
    c = min(out_shape[1], arr.shape[1])
    out[:r, :c] = arr[:r, :c]
    return out

def transpose2d(a):
    return np.ascontiguousarray(a.T)

def write_i32(f, arr, scale=1000.0):
    a = np.asarray(arr, dtype=np.float32)
    q = np.rint(a * scale).astype(np.int32, copy=False).reshape(-1)
    f.write(q.tobytes(order="C"))

# Build target vocab size from pair index logic used by runner
BASE_V = 256
V = BASE_V + pair_k
Vpad = ((V + 15) // 16) * 16
K2 = pair_k - pair_k1

with open(index_bin, "rb") as idxf:
    index_data = idxf.read()
pow2 = struct.unpack("<I", index_data[16:20])[0]

with open(env_file, "w", encoding="utf-8") as f:
    f.write(f"export DMODEL={D}\n")
    f.write(f"export NHEAD={H}\n")
    f.write(f"export NLAY={L}\n")
    f.write(f"export FFN={F}\n")
    f.write(f"export TMAX={TMAX}\n")
    f.write(f"export CKPT_HASH={os.path.basename(ckpt_file).split('_')[-1].split('.')[0]}\n")
    f.write(f"export DEFAULT_CKPT_FILE={ckpt_file}\n")

if os.path.exists(ckpt_file):
    print(f"[*] Checkpoint already exists: {ckpt_file}", flush=True)
    sys.exit(0)

print(f"[*] Writing compressed native checkpoint: {ckpt_file}", flush=True)

with open(ckpt_file, "wb") as f:
    f.write(struct.pack("<II", 0x43484452, 11))
    f.write(struct.pack("<II", pair_k1, K2))
    f.write(struct.pack("<IIIII", D, H, L, F, TMAX))

    id2pair_len = pair_k1 * 2
    id2pair2_len = K2 * 4
    f.write(index_data[24 : 24 + id2pair_len])
    f.write(index_data[24 + id2pair_len : 24 + id2pair_len + id2pair2_len])
    f.write(struct.pack("<I", pow2))
    f.write(index_data[24 + id2pair_len + id2pair2_len:])

    # Embeddings -> [Vpad, D]
    wte_src = load_np("model.embed_tokens.weight")
    wte = fit_2d(wte_src, (Vpad, D))
    write_i32(f, wte)

    # wpe -> zeros [TMAX, D]
    wpe = zeros((TMAX, D))
    write_i32(f, wpe)

    # Per-layer compression into small-core shapes
    for l in range(L):
        src_l = min(l, L_orig - 1)

        # input_layernorm.weight -> [Dhf_small]
        rms1 = fit_1d(load_np(f"model.layers.{src_l}.input_layernorm.weight"), Dhf_small)
        write_i32(f, rms1)

        # q/k/v/o -> target [Dhf_small, Dhf_small] after transpose like run3/run_full style
        q = load_np(f"model.layers.{src_l}.self_attn.q_proj.weight")
        k = load_np(f"model.layers.{src_l}.self_attn.k_proj.weight")
        v = load_np(f"model.layers.{src_l}.self_attn.v_proj.weight")
        o = load_np(f"model.layers.{src_l}.self_attn.o_proj.weight")

        q_t = fit_2d(transpose2d(q) if q is not None else None, (Dhf_small, Dhf_small))
        k_t = fit_2d(transpose2d(k) if k is not None else None, (Dhf_small, Dhf_small))
        v_t = fit_2d(transpose2d(v) if v is not None else None, (Dhf_small, Dhf_small))
        o_t = fit_2d(transpose2d(o) if o is not None else None, (Dhf_small, Dhf_small))

        write_i32(f, q_t)
        write_i32(f, k_t)
        write_i32(f, v_t)
        write_i32(f, o_t)

        rms2 = fit_1d(load_np(f"model.layers.{src_l}.post_attention_layernorm.weight"), Dhf_small)
        write_i32(f, rms2)

        gate = load_np(f"model.layers.{src_l}.mlp.gate_proj.weight")
        down = load_np(f"model.layers.{src_l}.mlp.down_proj.weight")

        gate_t = fit_2d(transpose2d(gate) if gate is not None else None, (F, Dhf_small))
        down_t = fit_2d(transpose2d(down) if down is not None else None, (Dhf_small, F))

        write_i32(f, gate_t)
        write_i32(f, down_t)

        print(f"[*] Layer {l+1}/{L} compressed", flush=True)

    norm = fit_1d(load_np("model.norm.weight"), D)
    write_i32(f, norm)

    lm_head = load_np("lm_head.weight")
    lm_t = fit_2d(transpose2d(lm_head) if lm_head is not None else None, (D, Vpad))
    write_i32(f, lm_t)

print("[*] Native compressed checkpoint done.", flush=True)
PY

echo "[*] Running zero-torch converter..."
HF_MODEL="$HF_MODEL" WORKDIR="$WORKDIR" INDEX_BIN="$INDEX_BIN" \
PAIR_K="$PAIR_K" PAIR_K1="$PAIR_K1" DMODEL="$DMODEL" NHEAD="$NHEAD" NLAY="$NLAY" FFN="$FFN" TMAX="$TMAX" \
ENV_FILE="$ENV_FILE" CKPT_FILE="$CKPT_FILE" \
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

echo "[*] Command: ${CMD[*]}"
exec "${CMD[@]}"
