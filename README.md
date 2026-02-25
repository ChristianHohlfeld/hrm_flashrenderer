# HRM FlashRenderer

**Applied to xAI - Member of Technical Staff Inference**  
*February 24, 2026*

This repository is my live portfolio, showcasing a high-performance LLM inference stack designed for efficiency and extreme environments.

## Why HRM-Flash?

This stack is built on three pillars of **vertical integration** and **architectural reliability**:

1.  **Full-Integrity Grounding (HRM Core):** Unlike standard vector-based RAG, the **HRM Core** (C++17) uses a deterministic, integer-only indexing engine. This ensures 100% control over the retrieval set, serving as a "Single Source of Truth" that prevents LLM hallucinations through explicit citation validation.
2.  **Native CUDA Optimization:** Instead of relying on heavy, generic frameworks, this project natively integrates a **custom SM75 FlashAttention kernel** (`csrc/flash_attn_sm75.cu`). It is specifically tuned for Turing-architecture GPUs (e.g., T4, 2080 Ti), implementing paged KV cache and optimized append operations directly.
3.  **Decoupled Knowledge & Compute:** The architecture strictly separates the knowledge base (Index) from the reasoning engine (LLM). This allows for hardware-agnostic scaling: run the same indexed data on **Zero-VRAM CPU setups** for reliability, or scale to **Multi-GPU Tensor Parallel clusters** for high-throughput production.

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
