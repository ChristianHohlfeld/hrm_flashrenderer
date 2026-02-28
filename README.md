# HRM FlashRenderer

## VRAM Reduction for Large Models on Legacy GPUs

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
# 1. Download example data (Shakespeare + Small LLM)
curl -L -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
curl -L -o model.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

# 2. Prepare and build the HRM index
hrm_core/build/hrm prep --input input.txt --out payloads.jsonl --cluster-size 200
hrm_core/build/hrm build --payloads payloads.jsonl --outdir ./model
```

### 2. Run inference

``` bash
CUDA_VISIBLE_DEVICES=0,1 python renderer/hrm_render.py \
   --model ./model \
   --hrm_bin hrm_core/build/hrm \
   --llm ./model.gguf \
   --prompt "What's the meaning of life?" \
   --n_gpu_layers 50
```

If this runs without crashing, your environment is correctly configured.

------------------------------------------------------------------------

## Custom FlashAttention Path (Optional)

Use this only if you have sufficient VRAM and CUDA properly configured.

``` bash
export HUGGING_FACE_HUB_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```
``` bash
# 1. Start the TP daemon (optional, for low latency)
# hrm-flash daemon --model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 --world 2

# 2. Run generate (uses 'model' dir created in build step)
hrm-flash generate \
   --hrm_model ./model \
   --llm_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 \
   --world 2 \
   --prompt "What's the meaning of life?"
```

> [!TIP]
> Use `--device cpu` to test the pipeline without CUDA. The CLI now performs early validation of HRM models to prevent silent hangs.


------------------------------------------------------------------------

## Benchmarking (Unified CLI)

For reproducible, signal-focused comparisons, use the unified benchmark CLI:

```bash
cd experiments
./benchmark_cli.sh -h
```

It prints a compact table directly in the console and ends with a winner line:

- `tok/s` (throughput)
- `run` (average run time)
- `total` (total time)
- `wins` (scenario wins)
- `🏆 WINNER`

### Common benchmark modes

```bash
# Quick matrix (small corpora, fast smoke test)
./benchmark_cli.sh --preset matrix --repeats 1 --steps 60 --lr-list 0.0003

# Decision mode (recommended default for tuning)
./benchmark_cli.sh --preset decision --repeats 2 --steps 80 --lr-list 0.0001,0.0003,0.001

# Larger-data mode (tiny/medium/large corpus mix)
./benchmark_cli.sh --preset large --repeats 2 --steps 120
```

### Key options

- `--preset matrix|decision|large`
- `--repeats <n>`
- `--steps <n>`
- `--lr-list <comma-separated>` (decimal format, e.g. `0.0001,0.0003,0.001`)
- `--include-beast 0|1`
- `--out <dir>`

Raw results are written to:

```text
experiments/bench_cli/raw.csv
```

------------------------------------------------------------------------

## Architecture Overview

-   Retrieval is deterministic and integer-based.
-   Retrieval runs on CPU/RAM/SSD (zero GPU VRAM usage).
-   The LLM operates only on bounded context.
-   Knowledge and compute are strictly separated.

This reduces VRAM pressure and allows operation near hardware limits.

------------------------------------------------------------------------

## Disclaimer

I'm Christian Heinrich Hohlfeld, B.Sc. Software Engineering.
Full honesty: I’m not a traditional CUDA kernel veteran or ninja. 
I guide AIs precisely towards my goals — and turn ideas into clean,
working, performant code very fast.

The underlying idea behind HRM, the architectural decisions and ideas,
and the papers that led to it are my own work.

I built and open-sourced **hrm_flashrenderer** to address a concrete
hardware constraint: running large models on legacy 11GB GPUs.

The implementation was developed using Agentic Coding. I defined the
software architecture and its constraints, system architecture, and verification
criteria. Large parts of the code --- including the SM75 FlashAttention
kernel --- were generated and iteratively refined using LLMs under my
direction.

This repository demonstrates that workflow applied to low-VRAM LLM
inference.

christianhohlfeld.com GitHub: ChristianHohlfeld
