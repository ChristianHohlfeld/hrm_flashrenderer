#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/hrm_flashrenderer}"
EXP_DIR="${EXP_DIR:-$REPO_DIR/experiments}"
HF_MODEL="${HF_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
MODE="${1:---chat}"

cd "$EXP_DIR"

echo "[*] Repo: $REPO_DIR"
echo "[*] Experiments: $EXP_DIR"
echo "[*] HF model: $HF_MODEL"
echo "[*] Mode: $MODE"
echo "[*] Forcing fresh native checkpoint regeneration..."

./run3.sh --force-new --hf "$HF_MODEL" "$MODE"

