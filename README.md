# HRM FlashRenderer

## Mission: Extreme VRAM Reduction for Large Models on Legacy GPUs

This repository presents my invention: the **HRM (Hash Retrieval
Model)**.

The core objective is to solve a strict hardware constraint: running
massive LLMs (32B+ parameters) on legacy, VRAM-constrained GPUs such as
**2×11 GB RTX 2080 Ti**.

Instead of trying to squeeze heavy matrix multiplications into limited
VRAM, the architecture shifts the burden of knowledge retrieval away
from the GPU.

------------------------------------------------------------------------

## Quick Start

The steps below provide a straightforward path to getting the system
running using the Local Renderer (GGUF + llama.cpp).

### 1. Clone and Install

``` bash
git clone https://github.com/ChristianHohlfeld/hrm_flashrenderer.git
cd hrm_flashrenderer

sudo apt install libsqlite3-dev build-essential cmake
python3 -m pip install -r requirements.prod.txt
python3 -m pip install -e .

make build
```

### 2. Build Index

``` bash
hrm_core/build/hrm prep --input your_data.txt --out payloads.jsonl
hrm_core/build/hrm build --payloads payloads.jsonl --outdir model_index
```

### 3. Run (2×11GB GPUs)

``` bash
CUDA_VISIBLE_DEVICES=0,1 python renderer/hrm_render.py   --model model_index   --hrm_bin hrm_core/build/hrm   --llm /path/to/model.gguf   --prompt "Your question"   --n_gpu_layers 50   --tensor_split 0.5 0.5
```

If this executes successfully, your environment is correctly configured.

------------------------------------------------------------------------

## Core Philosophy

### HRM Core & Context Bounding

The HRM Core selects context deterministically and with precision,
enabling operation under a strict prompt budget.\
This bounded context is a key enabler for running 30B+ models within 11
GB VRAM constraints.

### VRAM-Reduction Layer

The GGUF path currently serves as the practical Local Renderer optimized
for legacy hardware efficiency.

### Vision: Resonant Sparse Attention (RSA)

The long-term direction is to move beyond dense matrix multiplications
toward associative recall and sparse routing.\
Quantization techniques such as GPTQ-Int4 are transitional solutions
rather than the final architectural goal.

This architecture was realized through **Agentic Coding**: translating
conceptual design and strict architectural constraints into a
high-performance implementation using LLM-assisted development.

------------------------------------------------------------------------

## Advanced: Custom FlashAttention Path

For maximum performance with sufficient VRAM:

``` bash
hrm-flash generate   --hrm_model model_index   --llm_model models/Qwen2.5-32B-Instruct-GPTQ-Int4   --world 2   --prompt "Your question"
```

------------------------------------------------------------------------

## Why HRM-Flash?

1.  **Zero-VRAM Retrieval (HRM Core)**\
    Deterministic, integer-only indexing. Retrieval runs entirely on
    CPU/RAM/SSD.

2.  **Legacy-Tuned CUDA Optimization**\
    Custom SM75 FlashAttention kernel optimized for Turing GPUs (2080
    Ti, T4) with paged KV-cache and append optimizations.

3.  **Decoupled Knowledge & Compute**\
    Retrieval and reasoning are strictly separated. The GPU operates at
    its hardware limit without hosting retrieval overhead.

------------------------------------------------------------------------

## Reality Check

Running 32B+ models on 11GB GPUs remains an extreme edge case.\
It requires careful prompt budgeting and deliberate parameter tuning.

------------------------------------------------------------------------

## Troubleshooting

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

## Disclaimer

I'm Christian Heinrich Hohlfeld, B.Sc. Software Engineering.

I developed the underlying ideas, architectural direction, mathematical
constraints, and conceptual papers behind HRM.

I built and open-sourced **hrm_flashrenderer** to address a concrete
hardware constraint: running large models on legacy 11GB GPUs.

The implementation was created in close collaboration with AI systems:\
I defined the architecture, constraints, and verification criteria,
while LLMs generated and iteratively refined large parts of the code ---
including the custom SM75 FlashAttention kernel --- under my technical
guidance and review.

This hybrid model --- human-driven design and leadership combined with
AI-assisted code generation --- reflects my approach to modern systems
engineering.

christianhohlfeld.com\
GitHub: ChristianHohlfeld
