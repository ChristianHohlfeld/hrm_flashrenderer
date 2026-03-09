#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# run5.sh
# Low-VRAM big-model DeepSeek chat runner with client-side PHO remap.
#
# Goals:
#   - actual DeepSeek responses
#   - keep VRAM low enough to make larger models possible
#   - self-contained shell script
#   - use CUDA when available
#   - client-side PHO remap after generation
#
# Strategy:
#   - Hugging Face CausalLM
#   - auto-select largest GPU
#   - 4-bit bitsandbytes quantization if available
#   - CPU offload fallback when needed
#   - generation runs on CUDA; PHO remap stays client-side
#
# Usage:
#   ./run5.sh
#   HF_MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" ./run5.sh
#   HF_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" MAX_NEW_TOKENS=192 ./run5.sh
#   JSON=1 ./run5.sh
#   SYSTEM_PROMPT="Answer briefly." ./run5.sh
#
# Notes:
#   - This path prioritizes "big model, low VRAM, actually runs".
#   - Model cache/checkpoints are handled by Hugging Face cache.
# =============================================================================

need(){ command -v "$1" >/dev/null 2>&1 || { echo "FATAL: $1 not found"; exit 1; }; }
need nvidia-smi

PY="${PY:-python3}"
need "$PY"

: "${HF_MODEL:=deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
: "${MAX_NEW_TOKENS:=160}"
: "${TEMPERATURE:=0.7}"
: "${TOP_P:=0.95}"
: "${TOP_K:=50}"
: "${REPETITION_PENALTY:=1.08}"
: "${JSON:=0}"
: "${SYSTEM_PROMPT:=You are helpful, precise, and concise.}"
: "${MODEL_DTYPE:=auto}"         # auto|float16|bfloat16
: "${USE_4BIT:=1}"               # 1 preferred
: "${CPU_OFFLOAD:=1}"            # 1 enables low_cpu_mem_usage/device_map auto
: "${TRUST_REMOTE_CODE:=1}"
: "${WORKDIR:=$PWD}"
: "${LOG_DIR:=$WORKDIR/logs}"

mkdir -p "$LOG_DIR"

GPU_INDEX="${GPU_INDEX:-}"
if [[ -z "$GPU_INDEX" ]]; then
  GPU_INDEX="$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits | sort -t, -k2 -nr | head -n1 | cut -d, -f1 | tr -d ' ')"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
slug="$(echo "$HF_MODEL" | sed 's#[/:]#_#g')"
LOGFILE="$LOG_DIR/${slug}_${timestamp}.log"

echo "[*] HF_MODEL=$HF_MODEL" | tee "$LOGFILE"
echo "[*] GPU_INDEX=$GPU_INDEX" | tee -a "$LOGFILE"
echo "[*] MAX_NEW_TOKENS=$MAX_NEW_TOKENS" | tee -a "$LOGFILE"
echo "[*] USE_4BIT=$USE_4BIT CPU_OFFLOAD=$CPU_OFFLOAD" | tee -a "$LOGFILE"

CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PY" - <<'PY' 2>&1 | tee -a "$LOGFILE"
import json
import os
import sys
import subprocess
from typing import List, Tuple

HF_MODEL = os.environ.get("HF_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "160"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
TOP_K = int(os.environ.get("TOP_K", "50"))
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", "1.08"))
JSON_MODE = os.environ.get("JSON", "0") == "1"
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "You are helpful, precise, and concise.")
MODEL_DTYPE = os.environ.get("MODEL_DTYPE", "auto").lower()
USE_4BIT = os.environ.get("USE_4BIT", "1") == "1"
CPU_OFFLOAD = os.environ.get("CPU_OFFLOAD", "1") == "1"
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "1") == "1"

def pip_install(pkgs: List[str]) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)

def ensure_imports():
    missing = []
    for mod, pkg in [
        ("torch", "torch"),
        ("transformers", "transformers>=4.48.0"),
        ("accelerate", "accelerate"),
        ("huggingface_hub", "huggingface_hub"),
        ("safetensors", "safetensors"),
        ("sentencepiece", "sentencepiece"),
    ]:
        try:
            __import__(mod)
        except Exception:
            missing.append(pkg)
    if missing:
        pip_install(missing)

ensure_imports()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

bnb_available = False
if USE_4BIT:
    try:
        import bitsandbytes  # noqa: F401
        bnb_available = True
    except Exception:
        try:
            pip_install(["bitsandbytes"])
            import bitsandbytes  # noqa: F401
            bnb_available = True
        except Exception:
            bnb_available = False

# ---------------- client-side PHO remap ----------------
PHO_TABLE: List[Tuple[int, bytes]] = [
    (1, b"the"), (2, b"and"), (3, b"ing"), (4, b"tion"), (5, b"ment"), (6, b"ions"),
    (7, b"that"), (8, b"with"), (9, b"have"), (10, b"this"), (11, b"from"), (12, b"were"),
    (13, b"tion "), (14, b"ing "), (15, b" of "), (16, b" to "), (17, b" in "), (18, b" for "),
    (19, b"sch"), (20, b"th"), (21, b"sh"), (22, b"ch"), (23, b"ph"), (24, b"wh"), (25, b"qu"),
    (26, b"ck"), (27, b"ng"), (28, b"oo"), (29, b"ee"), (30, b"ea"), (31, b"ou"), (32, b"ai"),
    (33, b"ie"), (34, b"ei"), (35, b"und"), (36, b"der"), (37, b"die"), (38, b"nicht"),
    (39, b"ich"), (40, b"#include"), (41, b"return "), (42, b"static "), (43, b"const "),
    (44, b"struct "), (45, b"class "), (46, b"template"), (47, b"uint32_t"), (48, b"uint16_t"),
    (49, b"float "), (50, b"int "), (51, b"size_t"), (52, b"->"), (53, b"::"), (54, b"=="),
    (55, b"!="), (56, b">="), (57, b"<="), (58, b"&&"), (59, b"||"), (60, b"++"), (61, b"--"),
    (62, b"/*"), (63, b"*/"), (64, b"//"),
]
PHO_ESC = 0

PHO_ORDER = sorted(PHO_TABLE, key=lambda kv: (-len(kv[1]), kv[1], kv[0]))
PHO_CODES = {code for code, _ in PHO_TABLE}

def pho_encode_bytes(data: bytes) -> List[int]:
    out: List[int] = []
    i = 0
    n = len(data)
    while i < n:
        b = data[i]
        if b == PHO_ESC or b in PHO_CODES:
            out.extend([PHO_ESC, b])
            i += 1
            continue
        matched = False
        for code, pat in PHO_ORDER:
            L = len(pat)
            if i + L <= n and data[i:i+L] == pat:
                out.append(code)
                i += L
                matched = True
                break
        if not matched:
            out.append(b)
            i += 1
    return out

def pho_encode_text(text: str) -> List[int]:
    return pho_encode_bytes(text.encode("utf-8", errors="replace"))

def get_dtype():
    if MODEL_DTYPE == "float16":
        return torch.float16
    if MODEL_DTYPE == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32

dtype = get_dtype()

print(f"[*] torch={torch.__version__} cuda={torch.cuda.is_available()} bnb={bnb_available}", flush=True)
if torch.cuda.is_available():
    print(f"[*] visible cuda device count={torch.cuda.device_count()}", flush=True)
    print(f"[*] active device name={torch.cuda.get_device_name(0)}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=TRUST_REMOTE_CODE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_kwargs = {
    "trust_remote_code": TRUST_REMOTE_CODE,
    "low_cpu_mem_usage": True,
}

if CPU_OFFLOAD:
    model_kwargs["device_map"] = "auto"

if bnb_available:
    from transformers import BitsAndBytesConfig
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
else:
    model_kwargs["torch_dtype"] = dtype

model = AutoModelForCausalLM.from_pretrained(HF_MODEL, **model_kwargs)
model.eval()

def build_messages(user_text: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

def render_prompt(user_text: str):
    messages = build_messages(user_text)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"[SYSTEM]\\n{SYSTEM_PROMPT}\\n[USER]\\n{user_text}\\n[ASSISTANT]\\n"

def answer(user_text: str) -> str:
    prompt = render_prompt(user_text)
    inputs = tokenizer(prompt, return_tensors="pt")
    if not CPU_OFFLOAD:
        device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return text

print("[chat] real DeepSeek generation + client-side PHO remap. /quit to exit.", flush=True)
while True:
    try:
        user_text = input("> ").strip()
    except EOFError:
        break
    if not user_text:
        continue
    if user_text == "/quit":
        break

    try:
        text = answer(user_text)
        pho_ids = pho_encode_text(text)
        if JSON_MODE:
            payload = {
                "prompt": user_text,
                "response_text": text,
                "pho_ids": pho_ids,
            }
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(text)
            print(f"[pho_ids] {pho_ids}")
    except torch.cuda.OutOfMemoryError:
        print("FATAL: CUDA OOM. Try a smaller model, lower MAX_NEW_TOKENS, or keep USE_4BIT=1.", flush=True)
        break
    except Exception as e:
        print(f"FATAL: {type(e).__name__}: {e}", flush=True)
        break
PY
