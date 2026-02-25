# HRM FlashRenderer

## Extreme VRAM Reduction for Large Models on Legacy GPUs

HRM (Hash Retrieval Model) is a deterministic retrieval layer designed
to reduce effective context size before inference.

The goal of this project is practical: enable running large language
models (30B+ parameters) on legacy GPUs with limited VRAM (e.g. 2×11GB
RTX 2080 Ti).

The core idea is simple: move knowledge retrieval off the GPU and
strictly bound the prompt context before inference.

------------------------------------------------------------------------

## System Requirements

Tested on Ubuntu 22.04 / 24.04.

Required:

-   Python 3.10+
-   CUDA toolkit (only for custom FlashAttention path)
-   build-essential
-   cmake
-   libsqlite3-dev
-   python3-venv
-   python3-full

Install system dependencies:

``` bash
sudo apt update
sudo apt install -y build-essential cmake libsqlite3-dev python3-venv python3-full
```

------------------------------------------------------------------------

## Installation (Recommended: Virtual Environment)

Ubuntu 23.04+ uses an externally managed Python environment (PEP 668).
Do NOT install packages globally with pip.

### 1. Clone the repository

``` bash
git clone https://github.com/ChristianHohlfeld/hrm_flashrenderer.git
cd hrm_flashrenderer
```

### 2. Create and activate virtual environment

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with:

    (.venv)

### 3. Upgrade pip

``` bash
pip install --upgrade pip
```

### 4. Install Python dependencies

``` bash
pip install -r requirements.prod.txt
pip install -e .
```

### 5. Build core components

``` bash
make build
```

If `make build` finishes without errors, installation is complete.

------------------------------------------------------------------------

## Quick Start (Local Renderer Path -- Low VRAM)

This is the most reliable way to run the system on 2×11GB GPUs.

### 1. Build the retrieval index

``` bash
hrm_core/build/hrm prep --input your_data.txt --out payloads.jsonl
hrm_core/build/hrm build --payloads payloads.jsonl --outdir model_index
```

### 2. Run inference

``` bash
CUDA_VISIBLE_DEVICES=0,1 python renderer/hrm_render.py   --model model_index   --hrm_bin hrm_core/build/hrm   --llm /path/to/model.gguf   --prompt "Your question"   --n_gpu_layers 50   --tensor_split 0.5 0.5
```

If this runs without crashing, your environment is correctly configured.

------------------------------------------------------------------------

## Custom FlashAttention Path (Optional)

Use this only if you have sufficient VRAM and CUDA properly configured.

``` bash
hrm-flash generate   --hrm_model model_index   --llm_model models/Qwen2.5-32B-Instruct-GPTQ-Int4   --world 2   --prompt "Your question"
```

------------------------------------------------------------------------

## Architecture Overview

-   Retrieval is deterministic and integer-based.
-   Retrieval runs on CPU/RAM/SSD (zero GPU VRAM usage).
-   The LLM operates only on bounded context.
-   Knowledge and compute are strictly separated.

This reduces VRAM pressure and allows operation near hardware limits.

------------------------------------------------------------------------

## Troubleshooting

  --------------------------------------------------------------------------------
  Problem                                      Fix
  -------------------------------------------- -----------------------------------
  Could NOT find SQLite3                       `sudo apt install libsqlite3-dev`
                                               then `make build`

  PEP 668 externally-managed-environment       Use a virtual environment
                                               (`python3 -m venv .venv`)

  No space left on device                      Check `df -h`

  HRM query failed (code=2)                    Rebuild the index

  Model path does not exist                    Use an absolute path

  OOM                                          Reduce `--max_seq_len` or GPU
                                               layers

  hrm binary not found                         Set `--hrm_bin hrm_core/build/hrm`
  --------------------------------------------------------------------------------

------------------------------------------------------------------------

## Disclaimer

I'm Christian Heinrich Hohlfeld, B.Sc. Software Engineering.

The underlying idea behind HRM, the architectural decisions and ideas,
and the papers that led to it are my work.

I built and open-sourced **hrm_flashrenderer** to address a concrete
hardware constraint: running large models on legacy 11GB GPUs.

The implementation was developed using Agentic Coding. I defined the
mathematical constraints, system architecture, and verification
criteria. Large parts of the code --- including the SM75 FlashAttention
kernel --- were generated and iteratively refined using LLMs under my
direction.

This repository demonstrates that workflow applied to low-VRAM LLM
inference.

christianhohlfeld.com GitHub: ChristianHohlfeld
