# HRM FlashRenderer

Deterministic retrieval and low-VRAM LLM execution for legacy GPUs.

This repository has two separate goals:

1. **Practical bounded-context inference** with HRM retrieval, where retrieval/indexing stays off-GPU and the LLM only sees a tightly bounded prompt.
2. **Experimental CUDA engine work** under `experiments/`, where custom kernels, deterministic indexing, benchmark wrappers, and checkpointed training/chat flows are compared directly.

---

## What the repo contains

There are three distinct usage layers:

### 1. HRM core + renderer path
This is the practical path.

Use it when you want:
- deterministic retrieval,
- bounded prompt context,
- lower VRAM pressure,
- a reproducible local workflow.

Key parts:
- `hrm_core/build/hrm`
- `renderer/hrm_render.py`
- generated retrieval model directory such as `./model`

### 2. Optional `hrm-flash` path
This is the native multi-GPU / tensor-parallel path described in the current repo README.

Use it when you want:
- the direct Flash path,
- multi-GPU generation,
- CPU-only validation before touching CUDA,
- early HRM model validation from the CLI.

### 3. `experiments/` path
This is the custom-engine and benchmarking area.

Use it when you want:
- controlled benchmarks,
- engine-vs-engine comparisons,
- deterministic checkpoint naming,
- custom CUDA experiment runs,
- train / continue / chat loops for the experimental engines.

Important scripts:
- `experiments/benchmark_cli.sh`
- `experiments/run_llm_orig.sh`
- `experiments/runbeast.sh`

---

## What is already confirmed from the repo

The following are directly confirmed from the current public repo contents:

- top-level install flow uses `requirements.prod.txt`, `pip install -e .`, and `make build`
- low-VRAM renderer flow uses `hrm_core/build/hrm` and `renderer/hrm_render.py`
- optional Flash path uses `hrm-flash generate`
- benchmark flow uses `experiments/benchmark_cli.sh`
- benchmark presets documented in the repo are `matrix`, `decision`, and `large`
- raw benchmark CSV output is documented as `experiments/bench_cli/raw.csv`
- `run_llm_orig.sh` and `runbeast.sh` both implement deterministic checkpoint hashing and wrapper-level train / continue / chat handling

What is **not** claimed here:
- I am **not** claiming every CUDA path was executed end-to-end in this environment, because that requires a working NVIDIA/CUDA setup and internet access for cloning/building.
- This README is therefore written to minimize user-facing failure risk by tightening commands, separating paths clearly, and calling out where CUDA is required.

---

## Supported usage modes

### Mode A — Practical local renderer path
Recommended first.

This is the safest first-run path for most users.

### Mode B — `hrm-flash` path
Use only after Mode A works.

### Mode C — `experiments/` path
Use only after base install works and you intentionally want the custom-engine / benchmark workflow.

Do **not** start with `experiments/` unless your goal is explicit engine benchmarking or custom CUDA work.

---

## System requirements

The current repo README explicitly documents:
- Ubuntu 22.04 / 24.04
- Python 3.10+
- `build-essential`
- `cmake`
- `libsqlite3-dev`
- `python3-venv`
- `python3-full`
- CUDA toolkit only for the custom FlashAttention / experimental CUDA paths

### Base requirements

Install:

```bash
sudo apt update
sudo apt install -y build-essential cmake libsqlite3-dev python3-venv python3-full curl
```

Why `curl` is included here:
- it is used in the documented quick-start flow,
- it is required by the experiment wrapper scripts.

### CUDA requirements for experimental paths

Required for:
- `hrm-flash` CUDA usage,
- `experiments/run_llm_orig.sh`,
- `experiments/runbeast.sh`.

You need:
- NVIDIA driver working,
- CUDA toolkit installed,
- `nvcc` visible on `PATH`.

Check before using experimental paths:

```bash
nvidia-smi
nvcc --version
python3 --version
```

If `nvidia-smi` or `nvcc` fails, do **not** try the experimental CUDA paths yet.

---

## Clean installation

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

### 6. Verify the basic files/commands exist

Run these from the repo root:

```bash
test -f requirements.prod.txt
test -f Makefile
test -f renderer/hrm_render.py
test -d experiments
```

Optional quick command discovery:

```bash
python -c "import os; print('renderer/hrm_render.py exists:', os.path.exists('renderer/hrm_render.py'))"
```

If `make build` fails, stop there and fix the build before touching any inference or experiments.

---

## Recommended first-run order

Use this exact order.

1. Base install
2. HRM core build
3. Local renderer path
4. Optional `hrm-flash` CPU validation
5. Optional `hrm-flash` CUDA path
6. Benchmark CLI help
7. Experimental wrappers

This avoids the common mistake of going straight into the most fragile CUDA-heavy workflow.

---

# Part 1 — Practical local renderer path

This is the recommended first successful run.

## Step 1: download example data

```bash
curl -L -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
curl -L -o model.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

Confirm the files exist:

```bash
ls -lh input.txt model.gguf
```

## Step 2: prepare payloads and build the HRM index

```bash
hrm_core/build/hrm prep --input input.txt --out payloads.jsonl --cluster-size 200
hrm_core/build/hrm build --payloads payloads.jsonl --outdir ./model
```

Confirm outputs exist:

```bash
ls -lah payloads.jsonl model
```

## Step 3: run the renderer path

```bash
CUDA_VISIBLE_DEVICES=0,1 python renderer/hrm_render.py \
  --model ./model \
  --hrm_bin hrm_core/build/hrm \
  --llm ./model.gguf \
  --prompt "What's the meaning of life?" \
  --n_gpu_layers 50
```

### Notes

- This is the path the repo currently frames as the most reliable low-VRAM local route.
- `CUDA_VISIBLE_DEVICES=0,1` assumes two GPUs. Adjust it if needed.
- If you only have one usable GPU, try one GPU first by adapting the visible device list and GPU-layer count.

### If this fails

Check:

```bash
ls -lah hrm_core/build/hrm
python -V
ls -lah renderer/hrm_render.py
```

Most common failure causes here:
- `make build` was not completed,
- `model.gguf` did not download correctly,
- `./model` was not created by the HRM build step,
- GPU offload settings are too aggressive for the local machine.

---

# Part 2 — Optional `hrm-flash` path

Use this only after the renderer path works.

## Step 1: set the Hugging Face token if required

```bash
export HUGGING_FACE_HUB_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

## Step 2: CPU-only validation first

This is the safest check before CUDA:

```bash
hrm-flash generate \
  --device cpu \
  --hrm_model ./model \
  --llm_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 \
  --world 1 \
  --prompt "test"
```

If this fails, fix the pipeline before trying the CUDA path.

## Step 3: optional daemon example

```bash
# Optional for lower-latency repeated use
# hrm-flash daemon --model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 --world 2
```

## Step 4: generation path

```bash
hrm-flash generate \
  --hrm_model ./model \
  --llm_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 \
  --world 2 \
  --prompt "What's the meaning of life?"
```

### Notes

- The repo README explicitly says the CLI performs early validation of HRM models.
- `--world 2` assumes a 2-way configuration. Use a world size your system can actually support.
- Do not start here if the base renderer flow is still failing.

---

# Part 3 — Benchmarking

The repo already documents a unified benchmark CLI under `experiments/`.

This is the correct entrypoint for reproducible comparisons.

## Enter the experiments directory

```bash
cd experiments
```

## Show help first

```bash
./benchmark_cli.sh -h
```

If that fails, check:

```bash
ls -lah ./benchmark_cli.sh
chmod +x ./benchmark_cli.sh
./benchmark_cli.sh -h
```

## What benchmark CLI is for

Use it when you want:
- a compact benchmark table,
- winner summaries,
- controlled preset-based comparisons,
- raw CSV output.

The repo README currently documents these benchmark outputs:
- `tok/s`
- `run`
- `total`
- `wins`
- winner line

## Presets documented by the repo

### Quick matrix

```bash
./benchmark_cli.sh --preset matrix --repeats 1 --steps 60 --lr-list 0.0003
```

Use this for:
- smoke testing,
- quick regressions,
- fast sanity checks.

### Decision mode

```bash
./benchmark_cli.sh --preset decision --repeats 2 --steps 80 --lr-list 0.0001,0.0003,0.001
```

Use this for:
- tuning defaults,
- more credible comparisons,
- release candidate decisions.

### Large mode

```bash
./benchmark_cli.sh --preset large --repeats 2 --steps 120
```

Use this for:
- broader corpus-size coverage,
- larger comparison sweeps,
- stronger stability checks than a smoke run.

## Raw benchmark CSV

The repo documents raw results at:

```text
experiments/bench_cli/raw.csv
```

After a run, confirm it exists:

```bash
ls -lah bench_cli/raw.csv
```

## Recommended first benchmark command

```bash
./benchmark_cli.sh --preset matrix --repeats 1 --steps 20 --lr-list 0.0003
```

Reason: start smaller than the README defaults if you are only checking that the pipeline works.

---

# Part 4 — Experimental wrapper: `run_llm_orig.sh`

This is the original custom-engine wrapper.

It does more than “run one binary”.

From the script itself, it is responsible for:
- checking `nvcc`, `g++`, and `curl`,
- downloading Tiny Shakespeare if no local corpus exists,
- building a deterministic v7-style index,
- generating and compiling the custom CUDA engine,
- handling deterministic checkpoint hashing,
- handling wrapper-level train / continue / chat behavior.

## Before using it

Check:

```bash
cd experiments
nvcc --version
g++ --version
curl --version
```

If any of these fail, stop and fix the environment first.

## Show wrapper help

```bash
./run_llm_orig.sh --help
```

If the file is not executable:

```bash
chmod +x ./run_llm_orig.sh
./run_llm_orig.sh --help
```

## What the wrapper does automatically

Important behavior from the script:
- downloads Tiny Shakespeare if the default corpus file is missing,
- creates deterministic checkpoint names based on architecture parameters,
- auto-enters chat mode if a compatible checkpoint exists and no explicit mode is passed,
- auto-enters train mode if no checkpoint exists and no explicit mode is passed,
- supports `--force-new` for fresh starts.

That means users should not manually invent checkpoint names unless they know why.

## Safe first smoke run

```bash
./run_llm_orig.sh --train --steps 20 --batch 16 --seq 128 --measure
```

Why this exact command:
- low enough step count for a smoke test,
- explicit mode,
- explicit sequence length,
- explicit throughput output.

## Continue a compatible run

```bash
./run_llm_orig.sh --train --continue --steps 100
```

## Force a fresh run

```bash
./run_llm_orig.sh --force-new --train --steps 100
```

## Chat mode

```bash
./run_llm_orig.sh --chat
```

## Chat with an initial prompt

```bash
./run_llm_orig.sh --chat --chat_prompt "Hello"
```

## Important wrapper/engine arguments surfaced by the script

Documented in the wrapper help and script flow:
- `--force-new`
- `--train`
- `--chat`
- `--chat_prompt`
- `--continue`
- `--steps`
- `--batch`
- `--seq`
- `--gpus`
- `--measure`
- `--no_graph`
- `--lr`
- `--ckpt`

## Important note on checkpoint behavior

This wrapper intentionally hashes checkpoint names from architecture settings.

That is good.

It reduces the chance of:
- shape-mismatch loads,
- accidental resume into a different architecture,
- confusing mixed checkpoints.

---

# Part 5 — Experimental wrapper: `runbeast.sh`

This is the more aggressive experimental wrapper.

Like `run_llm_orig.sh`, it also:
- checks system tools,
- builds deterministic indexes,
- compiles a CUDA engine,
- manages deterministic checkpoints,
- supports train / continue / chat workflows.

It additionally includes a `--pho` mode in its wrapper logic.

## Before using it

Check:

```bash
cd experiments
nvcc --version
g++ --version
curl --version
```

## Show wrapper help

```bash
./runbeast.sh --help
```

If needed:

```bash
chmod +x ./runbeast.sh
./runbeast.sh --help
```

## Safe first Beast smoke run

```bash
./runbeast.sh --train --steps 20 --batch 16 --seq 128 --measure
```

## Continue a compatible Beast run

```bash
./runbeast.sh --train --continue --steps 100
```

## Force a fresh Beast run

```bash
./runbeast.sh --force-new --train --steps 100
```

## Beast chat mode

```bash
./runbeast.sh --chat
```

## Beast PhO mode

Use only intentionally and benchmark it separately:

```bash
./runbeast.sh --pho --train --steps 100 --measure
```

Do not mix PhO and non-PhO results in one unlabeled benchmark table.

## Important Beast note

“Beast” should be treated as a benchmark candidate, not an assumed winner.

Use `benchmark_cli.sh` to determine whether it is actually better on your target hardware.

---

# Part 6 — Recommended release validation sequence

This is the cleanest full validation order for users.

## A. Base install validation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.prod.txt
pip install -e .
make build
```

## B. Base file checks

```bash
test -f renderer/hrm_render.py
test -f experiments/benchmark_cli.sh
test -f experiments/run_llm_orig.sh
test -f experiments/runbeast.sh
```

## C. Renderer path validation

```bash
curl -L -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
curl -L -o model.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

hrm_core/build/hrm prep --input input.txt --out payloads.jsonl --cluster-size 200
hrm_core/build/hrm build --payloads payloads.jsonl --outdir ./model

CUDA_VISIBLE_DEVICES=0,1 python renderer/hrm_render.py \
  --model ./model \
  --hrm_bin hrm_core/build/hrm \
  --llm ./model.gguf \
  --prompt "What's the meaning of life?" \
  --n_gpu_layers 50
```

## D. `hrm-flash` CPU validation

```bash
hrm-flash generate \
  --device cpu \
  --hrm_model ./model \
  --llm_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 \
  --world 1 \
  --prompt "test"
```

## E. Benchmark CLI discovery

```bash
cd experiments
./benchmark_cli.sh -h
```

## F. Original experimental smoke run

```bash
./run_llm_orig.sh --train --steps 20 --batch 16 --seq 128 --measure
```

## G. Beast experimental smoke run

```bash
./runbeast.sh --train --steps 20 --batch 16 --seq 128 --measure
```

## H. Structured comparison

```bash
./benchmark_cli.sh --preset decision --repeats 2 --steps 80 --lr-list 0.0001,0.0003,0.001 --include-beast 1
```

This order is deliberate. It reduces the chance that users hit the hardest path first and conclude the repo is broken.

---

# Part 7 — Common failure points and fixes

## 1. `make build` fails

Fix build first. Do not continue.

Check:

```bash
python3 --version
pip --version
ls -lah Makefile requirements.prod.txt
```

## 2. `hrm_core/build/hrm` missing

That means the core build step did not complete.

Fix:

```bash
make build
ls -lah hrm_core/build/
```

## 3. `renderer/hrm_render.py` missing

That means the repo checkout is incomplete or you are in the wrong directory.

Check:

```bash
pwd
ls -lah renderer/
```

## 4. `hrm-flash` not found

That usually means installation did not complete correctly.

Check:

```bash
which hrm-flash
pip show hrm-flash 2>/dev/null || true
```

If not found, reinstall in the active venv:

```bash
pip install -e .
```

## 5. `benchmark_cli.sh` permission denied

Fix:

```bash
chmod +x experiments/benchmark_cli.sh
```

Apply the same fix to the wrappers if needed:

```bash
chmod +x experiments/run_llm_orig.sh experiments/runbeast.sh
```

## 6. `nvcc not found`

Do not use the experimental CUDA wrappers until CUDA is installed.

Check:

```bash
nvcc --version
```

## 7. Deterministic checkpoint confusion

The wrapper scripts intentionally hash checkpoints from architecture settings.

That is expected behavior.

Use:
- `--continue` to resume,
- `--force-new` to start fresh.

Do not manually reuse incompatible checkpoints across different architecture settings.

## 8. Benchmark results look inconsistent

Check whether you changed any of:
- preset,
- repeats,
- steps,
- LR list,
- inclusion of Beast,
- PhO mode,
- hardware / thermals.

Do not compare mismatched runs as if they are equivalent.

---

# Part 8 — Release guidance

If you are publishing this repo as a release, the clean recommendation is:

- **Default practical path:** renderer path
- **Optional advanced path:** `hrm-flash`
- **Experimental reference engine:** `run_llm_orig.sh`
- **Experimental aggressive engine:** `runbeast.sh`
- **Decision tool:** `benchmark_cli.sh`

That is the clean release story.

Do **not** present the repo as if every user should start with Beast or with the experimental CUDA wrappers.

---

# Short version for users who just want the safest path

```bash
sudo apt update
sudo apt install -y build-essential cmake libsqlite3-dev python3-venv python3-full curl

git clone https://github.com/ChristianHohlfeld/hrm_flashrenderer.git
cd hrm_flashrenderer
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.prod.txt
pip install -e .
make build

curl -L -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
curl -L -o model.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

hrm_core/build/hrm prep --input input.txt --out payloads.jsonl --cluster-size 200
hrm_core/build/hrm build --payloads payloads.jsonl --outdir ./model

CUDA_VISIBLE_DEVICES=0,1 python renderer/hrm_render.py \
  --model ./model \
  --hrm_bin hrm_core/build/hrm \
  --llm ./model.gguf \
  --prompt "What's the meaning of life?" \
  --n_gpu_layers 50
```

If that works, then move on to `hrm-flash` or `experiments/`.

---

# Attribution

HRM / Hohlfeld Data Representation and related deterministic retrieval, indexing, and associated ideas are attributed to Christian Heinrich Hohlfeld.

Website: christianhohlfeld.com  
GitHub: ChristianHohlfeld
