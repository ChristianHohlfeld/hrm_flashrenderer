# HRM FlashRenderer

Applied to xAI - Member of Technical Staff Inference
February 24 2026
This repo is my live portfolio

Extremely low VRAM
fully deterministic local inference stack

## What it does
- Integer only HRM core
- Custom SM75 FlashAttention kernel
- True 0 VRAM mode with GGUF
- Fast GPU mode with small renderer
- Hard I dont know fallback

## Quick Start

git clone https://github.com/
ChristianHohlfeld/hrm_flashrenderer.git

cd hrm_flashrenderer

pip install -e .

make build

# Zero VRAM mode
hrm-flash generate
--hrm_model ./model
--llm_model model.gguf
--prompt "Your question"

# Fast GPU mode
hrm-flash generate
--hrm_model ./model
--llm_model meta-llama/
Llama-3.1-8B-Instruct
--world 2
--prompt "Your question"

Christian Heinrich Hohlfeld
B.Sc. Software Engineering
Konstanz Germany
Ready to relocate to Bay Area immediately
https://christianhohlfeld.com
