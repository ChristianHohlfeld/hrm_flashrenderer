# HRM FlashRenderer

## Mission: Extreme VRAM Reduction for Large Models on Legacy GPUs

This repository presents my invention: the **HRM (Hash Retrieval
Model)**.

The core objective is to solve a strict hardware constraint: running
massive LLMs (32B+ parameters) on legacy, VRAM-constrained GPUs such as
**2×11 GB RTX 2080 Ti**.

Instead of trying to squeeze heavy matrix multiplications into tiny
VRAM, the architecture fundamentally shifts the burden of knowledge
retrieval away from the GPU entirely.

------------------------------------------------------------------------

# 🚀 Quick Start (Frictionless Path)

Copy, paste, build, run.

## 1. Clone & Install

``` bash
git clone https://github.com/ChristianHohlfeld/hrm_flashrenderer.git
cd hrm_flashrenderer

sudo apt install libsqlite3-dev build-essential cmake
python3 -m pip install -r requirements.prod.txt
python3 -m pip install -e .

make build
```

------------------------------------------------------------------------

## 2. Build Index

``` bash
hrm_core/build/hrm prep --input your_data.txt --out payloads.jsonl
hrm_core/build/hrm build --payloads payloads.jsonl --outdir model_index
```

------------------------------------------------------------------------

## 3. Run (2×11GB GPUs)

``` bash
CUDA_VISIBLE_DEVICES=0,1 python renderer/hrm_render.py   --model model_index   --hrm_bin hrm_core/build/hrm   --llm /path/to/model.gguf   --prompt "Your question"   --n_gpu_layers 50   --tensor_split 0.5 0.5
```

If this works, your system is correctly configured.

------------------------------------------------------------------------

# 🧠 Core Philosophy

### HRM Core & Context Bounding

The HRM Core selects context so precisely that we can operate under a
strict prompt budget.\
This bounded context is the real lever that makes 30B+ models feasible
on 11 GB VRAM.

### VRAM-Reduction Layer

The GGUF path is not a fallback --- it is the current realization of the
"Local Renderer" optimized for legacy hardware efficiency.

### Vision: Resonant Sparse Attention (RSA)

The long-term goal is to replace dense MatMuls with associative recall
and sparse routing.\
GPTQ-Int4 is only a temporary bridge --- the direction is deterministic,
memory-efficient attention.

This architecture was realized through **Agentic Coding**: translating
my conceptual paper and strict architectural constraints into
high-performance, bare-metal code using LLMs.

------------------------------------------------------------------------

# ⚙ Advanced: Custom FlashAttention Path

For maximum performance with sufficient VRAM:

``` bash
hrm-flash generate   --hrm_model model_index   --llm_model models/Qwen2.5-32B-Instruct-GPTQ-Int4   --world 2   --prompt "Your question"
```

------------------------------------------------------------------------

# 🧩 Why HRM-Flash?

1.  **Zero-VRAM Retrieval (HRM Core)**\
    Deterministic, integer-only indexing. Retrieval runs entirely on
    CPU/RAM/SSD.

2.  **Legacy-Tuned CUDA Optimization**\
    Custom SM75 FlashAttention kernel optimized for Turing GPUs (2080
    Ti, T4) with paged KV-cache and append optimizations.

3.  **Decoupled Knowledge & Compute**\
    Retrieval and reasoning are strictly separated. The GPU operates at
    its hardware limit --- nothing more.

------------------------------------------------------------------------

# ⚠ Reality Check

Running 32B+ models on 11GB GPUs is an extreme edge case.\
It requires strict prompt budgeting and careful parameter tuning.

------------------------------------------------------------------------

# 🛠 Troubleshooting

  -----------------------------------------------------------------------
  Problem                             Solution
  ----------------------------------- -----------------------------------
  Could NOT find SQLite3              sudo apt install libsqlite3-dev +
                                      make build

  No space left on device             Check `df -h`, avoid storing GGUF
                                      and GPTQ simultaneously

  HRM query failed (code=2)           Rebuild index
                                      (`hrm_core/build/hrm build ...`)

  Model path does not exist           Use absolute model path

  OOM in full stack                   Reduce `--max_seq_len`

  hrm binary not found                Set `--hrm_bin hrm_core/build/hrm`
                                      explicitly
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 👤 Disclaimer

I'm Christian Heinrich Hohlfeld, B.Sc. Software Engineering.

Full transparency: I am not a traditional CUDA kernel veteran. What I do
exceptionally well is combine **15+ years of software engineering
experience** with modern **Agentic Coding workflows** to transform
architectural ideas into clean, working, performant systems --- fast.

I built and open-sourced **hrm_flashrenderer** to solve a real physical
constraint: running massive models on legacy 11GB GPUs.\
The stack --- including the custom SM75 FlashAttention kernel --- was
generated and refined using LLMs under my strict architectural and
mathematical constraints.

I want to bring this direct, pragmatic engineering approach to **xAI**.

**Let's talk.**

christianhohlfeld.com\
GitHub: ChristianHohlfeld
