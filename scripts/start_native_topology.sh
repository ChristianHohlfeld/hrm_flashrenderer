#!/usr/bin/env bash
set -euo pipefail

# Starts three hrm-flash HTTP services for heterogeneous 4-GPU hosts:
# - nvlink_pair (world=2): two 11GB 2080 Ti cards connected via NVLink
# - solo_22gb   (world=1): standalone 22GB 2080 Ti
# - solo_3080   (world=1): standalone 3080 (10GB)
#
# Usage:
#   scripts/start_native_topology.sh <hrm_model_dir> [llm_model_dir|auto]
#
# Optional env vars:
#   BACKEND           default: deepseek_int8   (or torch_tp)
#   GPU_NVLINK_PAIR   default: auto (detect NVLink pair)
#   GPU_22GB          default: auto (largest remaining VRAM card)
#   GPU_3080          default: auto (smallest remaining VRAM card)
#   STRICT_GPU_TOPOLOGY default: 1 (fail if dynamic topology inference fails or manual mapping mismatches detection)
#   MODEL_PROFILE             default: max_vram_hetero
#   LLM_MODEL_SOLO_22GB      default: profile or <llm_model_dir arg>
#   LLM_MODEL_NVLINK_PAIR    default: profile or <llm_model_dir arg>
#   LLM_MODEL_SOLO_3080      default: profile or <llm_model_dir arg>
#   RECO_MODEL_SOLO_22GB     default: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
#   RECO_MODEL_NVLINK_PAIR   default: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
#   RECO_MODEL_SOLO_3080     default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
#   MODEL_BIN_SOLO_22GB      default: unset
#   MODEL_BIN_NVLINK_PAIR    default: unset
#   MODEL_BIN_SOLO_3080      default: unset
#   TOKENIZER_MODEL_SOLO_22GB   default: unset (falls back to per-service llm_model)
#   TOKENIZER_MODEL_NVLINK_PAIR default: unset
#   TOKENIZER_MODEL_SOLO_3080   default: unset
#   NATIVE_ENGINE_BIN         default: unset
#   NATIVE_STARTUP_TIMEOUT_S  default: 120
#   NATIVE_REQUEST_TIMEOUT_S  default: 180
#   PORT_SOLO_22GB    default: 8081
#   PORT_NVLINK_PAIR  default: 8082
#   PORT_SOLO_3080    default: 8083
#   MAX_SEQ_SOLO_22GB default: 4096
#   MAX_SEQ_NVLINK_PAIR default: 4096
#   MAX_SEQ_SOLO_3080 default: 3072
#   PREFILL_SOLO_22GB default: 768
#   PREFILL_NVLINK_PAIR default: 768
#   PREFILL_SOLO_3080 default: 384
#   Legacy fallback env:
#     MAX_SEQ_SOLO / MAX_SEQ_NVLINK, PREFILL_SOLO / PREFILL_NVLINK
#   MAX_NEW_TOKENS    default: 256
#   LOCAL_FILES_ONLY  default: 1   (set 0 to allow online model fetch)
#   LOG_DIR           default: ./.run/services
#   STARTUP_WAIT_TIMEOUT_S default: 240
#   STARTUP_POLL_INTERVAL_S default: 2

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <hrm_model_dir> [llm_model_dir|auto]" >&2
  exit 2
fi

HRM_MODEL="$1"
LLM_MODEL="${2:-auto}"
BACKEND="${BACKEND:-deepseek_int8}"
MODEL_PROFILE="${MODEL_PROFILE:-max_vram_hetero}"

RECO_MODEL_SOLO_22GB="${RECO_MODEL_SOLO_22GB:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}"
RECO_MODEL_NVLINK_PAIR="${RECO_MODEL_NVLINK_PAIR:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}"
RECO_MODEL_SOLO_3080="${RECO_MODEL_SOLO_3080:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"

use_profile=0
case "$LLM_MODEL" in
  ""|auto|max|max_vram|max_vram_hetero)
    use_profile=1
    ;;
esac

if [[ "$use_profile" == "1" ]]; then
  if [[ "$BACKEND" != "deepseek_int8" ]]; then
    echo "ERR: auto profile requires BACKEND=deepseek_int8. Pass explicit llm_model for backend=$BACKEND." >&2
    exit 2
  fi
  if [[ "$MODEL_PROFILE" != "max_vram_hetero" ]]; then
    echo "ERR: unsupported MODEL_PROFILE=$MODEL_PROFILE (supported: max_vram_hetero)." >&2
    exit 2
  fi
  LLM_MODEL_SOLO_22GB="${LLM_MODEL_SOLO_22GB:-$RECO_MODEL_SOLO_22GB}"
  LLM_MODEL_NVLINK_PAIR="${LLM_MODEL_NVLINK_PAIR:-$RECO_MODEL_NVLINK_PAIR}"
  LLM_MODEL_SOLO_3080="${LLM_MODEL_SOLO_3080:-$RECO_MODEL_SOLO_3080}"
else
  LLM_MODEL_SOLO_22GB="${LLM_MODEL_SOLO_22GB:-$LLM_MODEL}"
  LLM_MODEL_NVLINK_PAIR="${LLM_MODEL_NVLINK_PAIR:-$LLM_MODEL}"
  LLM_MODEL_SOLO_3080="${LLM_MODEL_SOLO_3080:-$LLM_MODEL}"
fi
MODEL_BIN_SOLO_22GB="${MODEL_BIN_SOLO_22GB:-}"
MODEL_BIN_NVLINK_PAIR="${MODEL_BIN_NVLINK_PAIR:-}"
MODEL_BIN_SOLO_3080="${MODEL_BIN_SOLO_3080:-}"
TOKENIZER_MODEL_SOLO_22GB="${TOKENIZER_MODEL_SOLO_22GB:-}"
TOKENIZER_MODEL_NVLINK_PAIR="${TOKENIZER_MODEL_NVLINK_PAIR:-}"
TOKENIZER_MODEL_SOLO_3080="${TOKENIZER_MODEL_SOLO_3080:-}"
NATIVE_ENGINE_BIN="${NATIVE_ENGINE_BIN:-}"
NATIVE_STARTUP_TIMEOUT_S="${NATIVE_STARTUP_TIMEOUT_S:-120}"
NATIVE_REQUEST_TIMEOUT_S="${NATIVE_REQUEST_TIMEOUT_S:-180}"

GPU_NVLINK_PAIR="${GPU_NVLINK_PAIR:-auto}"
GPU_22GB="${GPU_22GB:-auto}"
GPU_3080="${GPU_3080:-auto}"
STRICT_GPU_TOPOLOGY="${STRICT_GPU_TOPOLOGY:-1}"

PORT_SOLO_22GB="${PORT_SOLO_22GB:-8081}"
PORT_NVLINK_PAIR="${PORT_NVLINK_PAIR:-8082}"
PORT_SOLO_3080="${PORT_SOLO_3080:-8083}"

MAX_SEQ_SOLO_22GB="${MAX_SEQ_SOLO_22GB:-${MAX_SEQ_SOLO:-4096}}"
MAX_SEQ_NVLINK_PAIR="${MAX_SEQ_NVLINK_PAIR:-${MAX_SEQ_NVLINK:-4096}}"
MAX_SEQ_SOLO_3080="${MAX_SEQ_SOLO_3080:-${MAX_SEQ_SOLO:-3072}}"
PREFILL_SOLO_22GB="${PREFILL_SOLO_22GB:-${PREFILL_SOLO:-768}}"
PREFILL_NVLINK_PAIR="${PREFILL_NVLINK_PAIR:-${PREFILL_NVLINK:-768}}"
PREFILL_SOLO_3080="${PREFILL_SOLO_3080:-${PREFILL_SOLO:-384}}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
STARTUP_WAIT_TIMEOUT_S="${STARTUP_WAIT_TIMEOUT_S:-240}"
STARTUP_POLL_INTERVAL_S="${STARTUP_POLL_INTERVAL_S:-2}"
STARTUP_WAIT_TIMEOUT_S="${STARTUP_WAIT_TIMEOUT_S%.*}"
STARTUP_POLL_INTERVAL_S="${STARTUP_POLL_INTERVAL_S%.*}"
if [[ -z "${STARTUP_WAIT_TIMEOUT_S:-}" || "$STARTUP_WAIT_TIMEOUT_S" -lt 1 ]]; then
  STARTUP_WAIT_TIMEOUT_S=240
fi
if [[ -z "${STARTUP_POLL_INTERVAL_S:-}" || "$STARTUP_POLL_INTERVAL_S" -lt 1 ]]; then
  STARTUP_POLL_INTERVAL_S=2
fi

LOG_DIR="${LOG_DIR:-./.run/services}"
mkdir -p "$LOG_DIR"

detect_gpu_layout() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    return 1
  fi

  local detected
  detected="$(python3 - <<'PY'
import re
import subprocess
import sys

def run(cmd):
    return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)

try:
    mem_out = run(["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"])
except Exception:
    sys.exit(1)

mem = {}
for ln in mem_out.splitlines():
    ln = ln.strip()
    if not ln:
        continue
    parts = [p.strip() for p in ln.split(",")]
    if len(parts) < 2:
        continue
    if not parts[0].isdigit():
        continue
    try:
        mem[int(parts[0])] = int(parts[1])
    except Exception:
        continue

idxs = sorted(mem.keys())
if len(idxs) < 2:
    sys.exit(1)

matrix = {}
try:
    topo = run(["nvidia-smi", "topo", "-m"])
    header = None
    for raw in topo.splitlines():
        line = raw.strip()
        if not line:
            continue
        toks = line.split()
        if header is None and any(t.startswith("GPU") for t in toks):
            header = [t for t in toks if t.startswith("GPU")]
            continue
        if header and toks and toks[0].startswith("GPU"):
            row = toks[0]
            for j, col in enumerate(header):
                k = j + 1
                if k >= len(toks):
                    break
                matrix[(row, col)] = toks[k]
except Exception:
    matrix = {}

pairs = []
for i in idxs:
    for j in idxs:
        if i >= j:
            continue
        link = matrix.get((f"GPU{i}", f"GPU{j}")) or matrix.get((f"GPU{j}", f"GPU{i}")) or ""
        if not str(link).startswith("NV"):
            continue
        m = re.match(r"NV(\d+)", str(link))
        width = int(m.group(1)) if m else 1
        score = (100000 if abs(mem[i] - mem[j]) <= 2048 else 0) + min(mem[i], mem[j]) * 100 + width
        pairs.append((score, i, j))

if pairs:
    pairs.sort(reverse=True)
    _, p0, p1 = pairs[0]
else:
    # Strict requirement: NVLink pair must be explicitly detected.
    sys.exit(2)

remaining = [i for i in idxs if i not in (p0, p1)]
if len(remaining) >= 2:
    solo22 = max(remaining, key=lambda x: mem[x])
    solo3080 = min(remaining, key=lambda x: mem[x])
elif len(remaining) == 1:
    solo22 = remaining[0]
    others = [i for i in idxs if i != solo22]
    solo3080 = min(others, key=lambda x: mem[x]) if others else solo22
else:
    solo22 = max(idxs, key=lambda x: mem[x])
    others = [i for i in idxs if i != solo22]
    solo3080 = min(others, key=lambda x: mem[x]) if others else solo22

print(f"AUTO_GPU_NVLINK_PAIR={p0},{p1}")
print(f"AUTO_GPU_22GB={solo22}")
print(f"AUTO_GPU_3080={solo3080}")
PY
)" || return 1

  eval "$detected"
  return 0
}

normalize_pair() {
  local pair="$1"
  local a b
  IFS=',' read -r a b <<< "$pair"
  if [[ -z "${a:-}" || -z "${b:-}" ]]; then
    echo "$pair"
    return 0
  fi
  if (( a <= b )); then
    echo "${a},${b}"
  else
    echo "${b},${a}"
  fi
}

if ! detect_gpu_layout; then
  echo "ERR: failed to dynamically infer GPU topology (including NVLink pair)." >&2
  echo "Set GPU_NVLINK_PAIR/GPU_22GB/GPU_3080 explicitly only if you intentionally disable strict mode." >&2
  exit 1
fi

DETECTED_PAIR_NORM="$(normalize_pair "$AUTO_GPU_NVLINK_PAIR")"
if [[ "$GPU_NVLINK_PAIR" == "auto" ]]; then
  GPU_NVLINK_PAIR="$AUTO_GPU_NVLINK_PAIR"
else
  MANUAL_PAIR_NORM="$(normalize_pair "$GPU_NVLINK_PAIR")"
  if [[ "$STRICT_GPU_TOPOLOGY" == "1" && "$MANUAL_PAIR_NORM" != "$DETECTED_PAIR_NORM" ]]; then
    echo "ERR: GPU_NVLINK_PAIR=$GPU_NVLINK_PAIR mismatches detected NVLink pair=$AUTO_GPU_NVLINK_PAIR" >&2
    exit 1
  fi
fi

if [[ "$GPU_22GB" == "auto" ]]; then
  GPU_22GB="$AUTO_GPU_22GB"
elif [[ "$STRICT_GPU_TOPOLOGY" == "1" && "$GPU_22GB" != "$AUTO_GPU_22GB" ]]; then
  echo "ERR: GPU_22GB=$GPU_22GB mismatches detected 22GB slot GPU=$AUTO_GPU_22GB" >&2
  exit 1
fi

if [[ "$GPU_3080" == "auto" ]]; then
  GPU_3080="$AUTO_GPU_3080"
elif [[ "$STRICT_GPU_TOPOLOGY" == "1" && "$GPU_3080" != "$AUTO_GPU_3080" ]]; then
  echo "ERR: GPU_3080=$GPU_3080 mismatches detected smallest-remaining slot GPU=$AUTO_GPU_3080" >&2
  exit 1
fi

echo "[gpu-map] detected NVLink pair=$AUTO_GPU_NVLINK_PAIR solo_22gb=$AUTO_GPU_22GB solo_3080=$AUTO_GPU_3080 strict=$STRICT_GPU_TOPOLOGY"
echo "[gpu-map] final mapping nvlink_pair=$GPU_NVLINK_PAIR solo_22gb=$GPU_22GB solo_3080=$GPU_3080"

echo "[profile] backend=$BACKEND profile=$MODEL_PROFILE auto_profile=$use_profile"
echo "[profile] models: solo_22gb=$LLM_MODEL_SOLO_22GB nvlink_pair=$LLM_MODEL_NVLINK_PAIR solo_3080=$LLM_MODEL_SOLO_3080"
echo "[profile] seq/prefill: solo_22gb=${MAX_SEQ_SOLO_22GB}/${PREFILL_SOLO_22GB} nvlink_pair=${MAX_SEQ_NVLINK_PAIR}/${PREFILL_NVLINK_PAIR} solo_3080=${MAX_SEQ_SOLO_3080}/${PREFILL_SOLO_3080}"

start_service() {
  local name="$1"
  local devices="$2"
  local model="$3"
  local model_bin="$4"
  local tokenizer_model="$5"
  local world="$6"
  local port="$7"
  local max_seq="$8"
  local prefill="$9"

  echo "[start] $name: devices=$devices world=$world port=$port backend=$BACKEND model=$model model_bin=${model_bin:-<auto>}"
  serve_args=(
    --hrm_model "$HRM_MODEL" \
    --llm_model "$model" \
    --world "$world" \
    --host 0.0.0.0 \
    --port "$port" \
    --max_seq_len "$max_seq" \
    --prefill_chunk_size "$prefill" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_concurrent 1 \
    --backend "$BACKEND" \
    --native_startup_timeout_s "$NATIVE_STARTUP_TIMEOUT_S" \
    --native_request_timeout_s "$NATIVE_REQUEST_TIMEOUT_S"
  )
  if [[ -n "$model_bin" ]]; then
    serve_args+=(--model_bin "$model_bin")
  fi
  if [[ -n "$tokenizer_model" ]]; then
    serve_args+=(--tokenizer_model "$tokenizer_model")
  fi
  if [[ -n "$NATIVE_ENGINE_BIN" ]]; then
    serve_args+=(--native_engine_bin "$NATIVE_ENGINE_BIN")
  fi
  if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
    serve_args+=(--local_files_only)
  fi

  CUDA_VISIBLE_DEVICES="$devices" nohup hrm-flash serve "${serve_args[@]}" >"$LOG_DIR/${name}.log" 2>&1 &
  echo $! > "$LOG_DIR/${name}.pid"
}

health_ok() {
  local url="$1"
  if ! command -v curl >/dev/null 2>&1; then
    return 1
  fi
  local out
  out="$(curl -fsS --max-time 2 "$url" 2>/dev/null || true)"
  if [[ -z "$out" ]]; then
    return 1
  fi
  if command -v python3 >/dev/null 2>&1; then
    printf '%s' "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); raise SystemExit(0 if bool(d.get('ok')) else 1)" >/dev/null 2>&1
    return $?
  fi
  [[ "$out" == *"\"ok\":true"* || "$out" == *"\"ok\": true"* ]]
}

wait_service() {
  local name="$1"
  local port="$2"
  local url="http://127.0.0.1:${port}/v1/health"
  local waited=0
  echo "[wait] $name -> $url (timeout=${STARTUP_WAIT_TIMEOUT_S}s)"
  while (( waited < STARTUP_WAIT_TIMEOUT_S )); do
    if health_ok "$url"; then
      echo "[ready] $name"
      return 0
    fi
    sleep "$STARTUP_POLL_INTERVAL_S"
    waited=$((waited + STARTUP_POLL_INTERVAL_S))
  done

  echo "ERR: service '$name' did not become healthy within ${STARTUP_WAIT_TIMEOUT_S}s" >&2
  if [[ -f "$LOG_DIR/${name}.log" ]]; then
    echo "----- last 80 lines: $LOG_DIR/${name}.log -----" >&2
    tail -n 80 "$LOG_DIR/${name}.log" >&2 || true
    echo "----------------------------------------------" >&2
  fi
  return 1
}

start_service "solo_22gb" "$GPU_22GB" "$LLM_MODEL_SOLO_22GB" "$MODEL_BIN_SOLO_22GB" "$TOKENIZER_MODEL_SOLO_22GB" 1 "$PORT_SOLO_22GB" "$MAX_SEQ_SOLO_22GB" "$PREFILL_SOLO_22GB"
start_service "nvlink_pair" "$GPU_NVLINK_PAIR" "$LLM_MODEL_NVLINK_PAIR" "$MODEL_BIN_NVLINK_PAIR" "$TOKENIZER_MODEL_NVLINK_PAIR" 2 "$PORT_NVLINK_PAIR" "$MAX_SEQ_NVLINK_PAIR" "$PREFILL_NVLINK_PAIR"
start_service "solo_3080" "$GPU_3080" "$LLM_MODEL_SOLO_3080" "$MODEL_BIN_SOLO_3080" "$TOKENIZER_MODEL_SOLO_3080" 1 "$PORT_SOLO_3080" "$MAX_SEQ_SOLO_3080" "$PREFILL_SOLO_3080"

if command -v curl >/dev/null 2>&1; then
  wait_service "solo_22gb" "$PORT_SOLO_22GB"
  wait_service "nvlink_pair" "$PORT_NVLINK_PAIR"
  wait_service "solo_3080" "$PORT_SOLO_3080"
else
  echo "[warn] curl not found; skipping service health wait."
fi

cat <<EOF

Services started.
Logs: $LOG_DIR

Endpoints:
  solo_22gb   -> http://127.0.0.1:$PORT_SOLO_22GB/v1/health
  nvlink_pair -> http://127.0.0.1:$PORT_NVLINK_PAIR/v1/health
  solo_3080   -> http://127.0.0.1:$PORT_SOLO_3080/v1/health

EOF
