# HRM FlashRenderer

**Applied to xAI - Member of Technical Staff Inference**  
*February 24, 2026*

This repository is my live portfolio, showcasing a high-performance LLM inference stack designed for efficiency and extreme environments.

## The Core Philosophy
This stack focuses on **architectural reliability** and **resource efficiency** by decoupling knowledge from compute:

- **Decoupled Knowledge & Compute:** The **HRM Core** (C++17) handles retrieval as a deterministic process, while the LLM acts purely as a renderer. This allows tiny models to outperform giants in factual accuracy.
- **Deterministic Grounding:** Every output is validated against the source index. If the LLM halluzinates or misses a quote, the system detects it and falls back to a safe, extractive mode.
- **Custom SM75 FlashAttention:** Natively integrated CUDA kernels for Tensor Parallel inference, ensuring the hardware is utilized to its limit without heavy frameworks.
- **Hardware-Agnostic Scaling:** The same indexed knowledge base works across **Zero-VRAM (CPU)** and **High-Performance (GPU)** environments.

## Prerequisites
- **GPU:** NVIDIA SM75 (T4, RTX 20-series) or better for FlashAttention.
- **Software:** CUDA Toolkit, CMake, Python 3.10+.
- **Dependencies:** `pip install -r requirements.prod.txt`

## Quick Start

### 1. Installation
```bash
git clone https://github.com/ChristianHohlfeld/hrm_flashrenderer.git
cd hrm_flashrenderer

# Recommended: Use a virtual environment
# python3 -m venv venv && source venv/bin/activate

# Or install directly with --break-system-packages (for local dev)
python3 -m pip install -r requirements.prod.txt --break-system-packages
python3 -m pip install -e . --break-system-packages
make build
```

### 2. Prepare Your Data (Indexing)
Before rendering, you must index your knowledge base. Here is an example using the Shakespeare dataset:
```bash
# 1. Download example data (Shakespeare + Small LLM)
curl -L -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
curl -L -o model.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

# 2. Prepare and build the HRM index
hrm_core/build/hrm prep --input input.txt --out payloads.jsonl --cluster-size 200
hrm_core/build/hrm build --payloads payloads.jsonl --outdir ./model
```

### 3. Usage Examples

#### Zero VRAM Mode (CPU/GGUF)
Uses `llama-cpp-python` for a tiny footprint.
```bash
python3 renderer/hrm_render.py \
  --model ./model \
  --llm ./model.gguf \
  --n_gpu_layers 0
```

#### Fast GPU Mode (Tensor Parallel)
Uses the custom FlashAttention engine for HF models.
```bash
hrm-flash generate \
  --hrm_model ./model \
  --llm_model meta-llama/Llama-3.1-8B-Instruct \
  --world 2 \
  --prompt "Your question"
```

---

# Disclaimer
I’m Christian Heinrich Hohlfeld, B.Sc. Software Engineering.

**Full honesty:** I’m not a traditional CUDA kernel veteran or ninja. What I do really well is guide AI precisely towards my goals — and turn ideas into clean, working, performant code very fast.

I built and open-sourced **hrm_flashrenderer** to demonstrate this approach, featuring a custom SM75 FlashAttention kernel with paged KV + append.

I want to bring this direct, pragmatic way of working to **xAI**. Ready to relocate to Bay Area / Seattle tomorrow.

**Let’s talk.**

[christianhohlfeld.com](https://christianhohlfeld.com) | [GitHub: ChristianHohlfeld](https://github.com/ChristianHohlfeld)
