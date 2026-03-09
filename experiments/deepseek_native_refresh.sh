#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/hrm_flashrenderer}"
EXP_DIR="${EXP_DIR:-$REPO_DIR/experiments}"

# Usage:
#   ./deepseek_native_run.sh
#   ./deepseek_native_run.sh deepseek-ai/DeepSeek-R1-Distill-Llama-8B
#   GPUS=2 TMAX_OVERRIDE=2048 SEQ=64 MODE=--chat ./deepseek_native_run.sh deepseek-ai/DeepSeek-R1-Distill-Llama-8B
#   HF_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" ./deepseek_native_run.sh
#
# Optional env vars:
#   HF_MODEL         HF model repo id
#   MODE             --chat | --train
#   GPUS             number of GPUs to request during runtime (default: 2)
#   SEQ              runtime sequence length, e.g. 64 or 128
#   TMAX_OVERRIDE    export TMAX before running, e.g. 1024/2048/4096
#   LOG_DIR          directory for logs (default: ./logs)
#   FORCE_NEW        1 to force fresh checkpoint regeneration (default: 1)
#   NO_GRAPH         1 to append --no_graph
#
# Examples:
#   ./deepseek_native_run.sh
#   HF_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" TMAX_OVERRIDE=2048 SEQ=64 ./deepseek_native_run.sh
#   HF_MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" GPUS=2 TMAX_OVERRIDE=1024 SEQ=32 NO_GRAPH=1 ./deepseek_native_run.sh

HF_MODEL="${HF_MODEL:-${1:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}}"
MODE="${MODE:---chat}"
GPUS="${GPUS:-2}"
SEQ="${SEQ:-}"
TMAX_OVERRIDE="${TMAX_OVERRIDE:-}"
LOG_DIR="${LOG_DIR:-$EXP_DIR/logs}"
FORCE_NEW="${FORCE_NEW:-1}"
NO_GRAPH="${NO_GRAPH:-0}"

mkdir -p "$LOG_DIR"
cd "$EXP_DIR"

slug="$(echo "$HF_MODEL" | sed 's#[/:]#_#g')"
ts="$(date +%Y%m%d_%H%M%S)"
logfile="$LOG_DIR/${slug}_${ts}.log"

cmd=( ./run3.sh )
if [[ "$FORCE_NEW" == "1" ]]; then
  cmd+=( --force-new )
fi
cmd+=( --hf "$HF_MODEL" "$MODE" --gpus "$GPUS" )
if [[ -n "$SEQ" ]]; then
  cmd+=( --seq "$SEQ" )
fi
if [[ "$NO_GRAPH" == "1" ]]; then
  cmd+=( --no_graph )
fi

echo "[*] Repo: $REPO_DIR" | tee "$logfile"
echo "[*] Experiments: $EXP_DIR" | tee -a "$logfile"
echo "[*] HF model: $HF_MODEL" | tee -a "$logfile"
echo "[*] Mode: $MODE" | tee -a "$logfile"
echo "[*] GPUs: $GPUS" | tee -a "$logfile"
if [[ -n "$SEQ" ]]; then
  echo "[*] Seq: $SEQ" | tee -a "$logfile"
fi
if [[ -n "$TMAX_OVERRIDE" ]]; then
  echo "[*] TMAX override: $TMAX_OVERRIDE" | tee -a "$logfile"
fi
if [[ "$NO_GRAPH" == "1" ]]; then
  echo "[*] no_graph: enabled" | tee -a "$logfile"
fi
echo "[*] Log: $logfile" | tee -a "$logfile"
echo "[*] Command: ${cmd[*]}" | tee -a "$logfile"

if [[ -n "$TMAX_OVERRIDE" ]]; then
  TMAX="$TMAX_OVERRIDE" "${cmd[@]}" 2>&1 | tee -a "$logfile"
else
  "${cmd[@]}" 2>&1 | tee -a "$logfile"
fi
