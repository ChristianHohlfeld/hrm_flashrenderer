# HRM FlashRenderer

Deterministic retrieval plus low-VRAM rendering for running larger language models on legacy GPUs, with an optional native tensor-parallel path for multi-GPU setups.

This project centers on **HRM (Hash Retrieval Model)**: a deterministic retrieval layer that reduces effective context size before inference so the LLM only sees a tightly bounded prompt. The practical goal is clear: make larger models usable on older hardware such as **RTX 2080 Ti-class GPUs**, especially in constrained multi-GPU environments.

---

## What this repository is for

`hrm_flashrenderer` is built around a simple systems idea:

- keep retrieval off the GPU
- keep prompt context bounded
- keep compute and knowledge separate
- make old hardware useful for modern inference workloads

That leads to two complementary execution paths:

1. **Low-VRAM local renderer path**  
   The most practical and reliable route for constrained systems such as **2×11 GB RTX 2080 Ti**.

2. **Native tensor-parallel Flash path**  
   A higher-performance path using the project's custom runtime and custom FlashAttention implementation, intended for setups with enough VRAM and a working CUDA toolchain.

---

## Core ideas

### Deterministic retrieval
HRM retrieval is integer-based and deterministic. For the same query, the system returns the same sources. It is intentionally not a vector database workflow.

### Strict separation of knowledge and compute
The retrieval/index side runs on CPU, RAM, and storage. The LLM side only receives a bounded context window.

### Legacy GPU practicality
The stack is explicitly aimed at making hardware like **Turing / SM75 GPUs** still useful.

### Vertical integration
The repository combines:
- HRM indexing and retrieval
- rendering/inference orchestration
- optional tensor-parallel runtime
- a custom CUDA attention path

---

## Execution paths

## 1) Low-VRAM renderer path

This is the recommended starting point and the path most clearly positioned as practical for older GPUs.

It uses:

- `renderer/hrm_render.py`
- the built HRM binary
- a GGUF model
- llama.cpp-style GPU layer offload / tensor split workflows

This path is described in the repo as the **most reliable** or **most practical** route for 2×11 GB systems.

Typical use case:
- constrained local workstation
- older GPUs
- larger model experimentation under hard VRAM limits
- predictable fallback when the full native stack is too memory-hungry

### Example
```bash
CUDA_VISIBLE_DEVICES=0,1 python renderer/hrm_render.py   --model ./model   --hrm_bin hrm_core/build/hrm   --llm ./model.gguf   --prompt "What's the meaning of life?"   --n_gpu_layers 50
```

---

## 2) Native Flash / tensor-parallel path

This is the higher-performance path.

It uses:

- `hrm-flash`
- the HRM model directory
- a Hugging Face model
- tensor parallelism
- the custom attention/runtime stack

The public repo material presents this as the path for users with sufficient VRAM and a correctly configured CUDA environment.

### Example
```bash
hrm-flash generate   --hrm_model ./model   --llm_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4   --world 2   --prompt "What's the meaning of life?"
```

### CPU validation mode
For pipeline testing without CUDA:
```bash
hrm-flash generate   --device cpu   --hrm_model ./model   --llm_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4   --world 2   --prompt "Test prompt"
```

---

## Hardware targets

The repository is explicitly motivated by **legacy GPUs with tight VRAM budgets**, especially **RTX 2080 Ti-class systems**.

Representative configurations:

### Practical baseline
- 2× RTX 2080 Ti (11 GB)
- Ubuntu 22.04 or 24.04
- enough system RAM for indexing and model handling
- CUDA toolkit for native Flash path

### Extended custom setup
- 4× RTX 2080 Ti 22 GB mod
- workstation/server board with enough PCIe lanes and power delivery
- Linux + CUDA
- strong thermal and power management

Important: multi-GPU VRAM remains **distributed**, not magically unified.

---

## Tensor parallel support

The repo's native runtime is intended to support multi-GPU tensor parallel execution.

Documented CLI examples prominently use `--world 2`, but the internal direction of the project is clearly toward multi-GPU tensor-parallel operation rather than single-box, single-GPU-only usage.

Use cases:
- larger models than fit on one card
- reduced per-device memory pressure
- optional daemon/server workflows for lower-latency repeated inference

Example daemon:
```bash
hrm-flash daemon --world 2
```

Example server:
```bash
hrm-flash serve --world 4
```

---

## Custom FlashAttention path

One of the defining parts of this repository is the custom attention stack, described in the repo materials as a self-written CUDA/SM75-oriented path.

The stated intent is to support **Turing GPUs such as RTX 2080 Ti and T4**, where many modern off-the-shelf attention kernels increasingly focus on newer architectures.

This path is relevant when:
- you want to squeeze more out of legacy hardware
- you control the full software stack
- generic modern kernels are a poor fit for Turing-era cards

This path is not presented as universal plug-and-play infrastructure for every machine and every model combination.

---

## System requirements

Tested on:
- Ubuntu 22.04
- Ubuntu 24.04

Required:
- Python 3.10+
- `build-essential`
- `cmake`
- `libsqlite3-dev`
- `python3-venv`
- `python3-full`
- CUDA toolkit for the custom FlashAttention/native path

Install system dependencies:
```bash
sudo apt update
sudo apt install -y build-essential cmake libsqlite3-dev python3-venv python3-full
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/ChristianHohlfeld/hrm_flashrenderer.git
cd hrm_flashrenderer
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade pip
```bash
pip install --upgrade pip
```

### 4. Install Python dependencies
```bash
pip install -r requirements.prod.txt
pip install -e .
```

### 5. Build core components
```bash
make build
```

If `make build` completes successfully, the repository is installed.

---

## Quick start

## A. Build an index

```bash
curl -L -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
curl -L -o model.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

hrm_core/build/hrm prep --input input.txt --out payloads.jsonl --cluster-size 200
hrm_core/build/hrm build --payloads payloads.jsonl --outdir ./model
```

## B. Run the low-VRAM renderer path

```bash
CUDA_VISIBLE_DEVICES=0,1 python renderer/hrm_render.py   --model ./model   --hrm_bin hrm_core/build/hrm   --llm ./model.gguf   --prompt "What's the meaning of life?"   --n_gpu_layers 50
```

## C. Run the native Flash path

```bash
export HUGGING_FACE_HUB_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX

hrm-flash generate   --hrm_model ./model   --llm_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4   --world 2   --prompt "What's the meaning of life?"
```

---

## Benchmarking

The repo includes a unified benchmark CLI under `experiments`.

Help:
```bash
cd experiments
./benchmark_cli.sh -h
```

Common modes:
```bash
./benchmark_cli.sh --preset matrix --repeats 1 --steps 60 --lr-list 0.0003
./benchmark_cli.sh --preset decision --repeats 2 --steps 80 --lr-list 0.0001,0.0003,0.001
./benchmark_cli.sh --preset large --repeats 2 --steps 120
```

Reported metrics include:
- throughput (`tok/s`)
- average run time
- total time
- wins by scenario
- winner summary

Raw results are written to:
```text
experiments/bench_cli/raw.csv
```

---

## Architecture overview

At a high level:

1. source data is prepared into payloads
2. the HRM index is built
3. retrieval runs off-GPU
4. only bounded context is handed to the model
5. rendering/inference happens through either:
   - the low-VRAM renderer path, or
   - the native Flash tensor-parallel path

This makes the project less about generic chatbot scaffolding and more about **systems architecture for constrained inference**.

---

## Why this matters

Most current inference tooling trends toward:
- newer GPU generations
- larger VRAM assumptions
- BF16-first expectations
- kernels tuned for Ampere and newer

`hrm_flashrenderer` pushes in the opposite direction:
- practical use of older GPUs
- careful VRAM budgeting
- CPU/GPU role separation
- deterministic retrieval before inference
- custom low-level tuning where needed

That makes it especially interesting for:
- researchers with older workstations
- constrained local inference setups
- hardware-frugal experimentation
- custom model-serving stacks on legacy GPUs

---

## Troubleshooting

### SQLite not found
Install the package and rebuild:
```bash
sudo apt install libsqlite3-dev
make build
```

### No space left on device
Check storage usage and avoid keeping multiple large model formats unnecessarily:
```bash
df -h
```

### HRM query failed
Rebuild the index:
```bash
hrm_core/build/hrm build --payloads payloads.jsonl --outdir model_index
```

### Model path does not exist
Use absolute paths and verify the file exists.

### OOM in the full native stack
Reduce sequence length or move to the low-VRAM path.

### `hrm` binary not found
Pass it explicitly:
```bash
--hrm_bin hrm_core/build/hrm
```

---

## Honest limitations

This repository is ambitious and practical, but it should be described honestly.

- The **low-VRAM path** is the clearest practical entry point.
- The **native Flash/tensor-parallel path** is more demanding and assumes a working CUDA/toolchain environment.
- Large models on 11 GB-class GPUs still operate close to hardware limits.
- Success depends on prompt budgeting, model choice, quantization format, and system tuning.
- This is a **custom stack**, not a mainstream one-click inference package.

That is part of the point: the project is about making constrained hardware viable through architecture and implementation choices.

---

## Attribution

Inventor / author attribution requested by the repository owner:

**Christian Heinrich Hohlfeld**  
Konstanz, Germany  
https://christianhohlfeld.com  
ORCID: 0009-0003-6634-9045

Requested attribution for generated code and related work:
**© 2026 Christian Heinrich Hohlfeld (Konstanz, Germany), christianhohlfeld.com, ORCID 0009-0003-6634-9045**

---

## Suggested positioning

A concise way to describe the project:

> HRM FlashRenderer is a deterministic retrieval-plus-rendering stack for low-VRAM local LLM inference, designed to make larger models usable on legacy GPUs through bounded context, off-GPU retrieval, and an optional native tensor-parallel FlashAttention path.

