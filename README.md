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

# Disclaimer
I’m Christian Heinrich Hohlfeld, B.Sc. Software Engineering.
Full honesty: I’m not a traditional CUDA kernel veteran or ninja.
What I do really well is guide AI precisely towards my goals — and turn ideas into clean, working, performant code very fast.
Proof: In just a few days I built & open-sourced
hrm_flashrenderer → 
https://github.com/ChristianHohlfeld/hrm_flashrenderer 
(including my own SM75 FlashAttention kernel with paged KV + append).
I want to bring this direct, pragmatic way of working to xAI.
Ready to relocate to Bay Area / Seattle tomorrow.
Let’s talk.
Christian Heinrich Hohlfeld
https://christianhohlfeld.com
GitHub: ChristianHohlfeld
