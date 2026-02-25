# HRM FlashRenderer

**Deterministic Retrieval + Rendering Stack for Low‑VRAM Local
Inference**

This repository presents my invention: the HRM (Hash Retrieval Model).
It demonstrates the complete architecture, including a self-written CUDA
kernel.

## Why HRM-Flash?

This stack is built on three pillars of **vertical integration** and
**architectural reliability**:

1.  **Full-Integrity Grounding (HRM Core):** The C++17 core uses
    deterministic, integer-only indexing. No vectors, no randomness ---
    always the same sources for the same query.
2.  **Native CUDA Optimization:** Custom SM75 FlashAttention kernel with
    paged KV-cache and optimized append operations, specifically tuned
    for Turing GPUs (2080 Ti, T4, etc.).
3.  **Decoupled Knowledge & Compute:** Knowledge base (index) and
    reasoning engine are strictly separated --- enabling scaling from
    zero‑VRAM CPU setups to multi‑GPU tensor parallel deployments.

## Reality Check

This stack is **not** plug & play for every scenario.

-   The **full native stack** (`hrm-flash` + tensor parallel + custom
    kernel) delivers maximum performance but requires more VRAM.
-   The **low‑VRAM fallback** (`renderer/hrm_render.py` + GGUF +
    llama.cpp `tensor_split`) is the most practical path for 2×11GB GPUs
    (e.g., 2×2080 Ti).
-   Large models (32B+) on 11GB per GPU operate at hardware limits ---
    you must work with prompt budgeting and adjusted parameters.

## Two Execution Paths

### 1. Full Native Stack (with sufficient VRAM)

Uses my tensor-parallel engine and my custom FlashAttention kernel.

``` bash
hrm-flash generate   --hrm_model model_index   --llm_model models/Qwen2.5-32B-Instruct-GPTQ-Int4   --world 2   --prompt "Your question"
```

### 2. Low‑VRAM Fallback (for 2×11GB GPUs)

Uses GGUF + llama.cpp with tensor_split.

``` bash
python renderer/hrm_render.py   --model model_index   --hrm_bin hrm_core/build/hrm   --llm /path/to/model.gguf   --n_gpu_layers 50   --tensor_split 0.5 0.5
```

## Requirements

### System:

Ubuntu / Linux\
CUDA Toolkit (for the full stack)

``` bash
sudo apt install libsqlite3-dev build-essential cmake
```

### Python:

``` bash
python3 -m pip install -r requirements.prod.txt
python3 -m pip install -e .
```

### Build:

``` bash
make build
```

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

  OOM in full stack                   Reduce `--max_seq_len` to 512 or
                                      lower

  hrm binary not found                Explicitly set
                                      `--hrm_bin hrm_core/build/hrm`
  -----------------------------------------------------------------------

## Quick Start (Low‑VRAM Path)

``` bash
# Build index
hrm_core/build/hrm prep --input your_data.txt --out payloads.jsonl
hrm_core/build/hrm build --payloads payloads.jsonl --outdir model_index

# Run
CUDA_VISIBLE_DEVICES=0,1 python renderer/hrm_render.py   --model model_index   --hrm_bin hrm_core/build/hrm   --llm /path/to/model.gguf   --prompt "Your question"   --n_gpu_layers 50   --n_ctx 4096   --max_tokens 512   --top_k 4
```

## Disclaimer

I am Christian Heinrich Hohlfeld, B.Sc. Software Engineering.\
Full honesty: I am not a traditional CUDA ninja. What I am truly good at
is steering AI precisely and turning ideas into clean, performant code
quickly.\
I built and open-sourced hrm_flashrenderer to demonstrate exactly this
way of working --- including my own SM75 FlashAttention kernel.\
I want to bring this direct, pragmatic way of building to xAI. Ready for
Bay Area / Seattle.

Let's talk.\
christianhohlfeld.com \| GitHub
