# HRM FlashRenderer

Native HRM retrieval + native DeepSeek inference for heterogeneous multi-GPU hosts.

Mainline scope is intentionally strict:
- no container requirement
- no Torch runtime on the production path
- backend fixed to `deepseek_int8`
- supported generation models fixed to:
  - `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
  - `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` (recommended with `MODEL_QUANT=q4`)

## Platform And Prerequisites

Target platform: Ubuntu server with NVIDIA GPUs.

Required system tools:
- `bash`
- `python3` (3.10-3.12)
- `curl`
- `nvidia-smi`
- `nvcc`

Required only when bootstrap/build is enabled:
- `cmake`
- `ctest`
- C/C++ toolchain (`build-essential` on Ubuntu)

Python dependencies are kept lean:
- `requirements.prod.txt`: `numpy`, `huggingface_hub`, `transformers`, `safetensors`
- `requirements.server.txt`: `fastapi`, `uvicorn`, `pydantic`

## First-User Path (Recommended)

If you already have an HRM index at `./model_index` (`router_index.bin` + `index.sqlite`), run:

```bash
bash scripts/prod_live_e2e.sh ./model_index "Bitte antworte auf Deutsch in 3 kurzen Bulletpoints: 1) Stack-Status 2) Kernaussage 3) Route-Hinweis."
```

What this command does end-to-end:
1. optional bootstrap (`RUN_BOOTSTRAP=1` default)
2. production preflight
3. native stack startup
4. health checks
5. strict mode checks (`mixed`, `retrieval`, `deepseek_only`)
6. deterministic repetition checks (`E2E_DETERMINISM_RUNS=3` default)
7. hard failure if mode guarantees are violated

Success marker:
- `E2E PASS: DeepSeek stack started and mode checks passed.`

Default E2E assumptions:
- `EXPECTED_GPUS=3`
- `TOPOLOGY_MODE=max_model_fast`
- `ROUTER_MODE=mixed`
- `RUN_MODE_MATRIX=1`

## Hardware Presets (Easy Start)

One-command startup with fixed profiles:

```bash
# Usage: scripts/start_easy.sh <hrm_model_dir> [A|B|C|D] [q8|q4]
bash scripts/start_easy.sh ./model_index A q8
```

Preset matrix:
- `A`: `22GB + 11GB + 11GB`, no NVLink required, no dedicated 3080 lane
- `B`: `22GB + 11GB + 11GB + 10GB`, no NVLink required, dedicated 3080 lane enabled
- `C`: `22GB + 11GB + 11GB + 10GB`, NVLink required on 11GB pair, dedicated 3080 lane enabled
- `D`: `22GB + 22GB + 11GB + 11GB`, no dedicated 3080 lane

Quant defaults:
- `q8` -> defaults to 32B (`DeepSeek-R1-Distill-Qwen-32B`)
- `q4` -> defaults to 70B (`DeepSeek-R1-Distill-Llama-70B`)

## Topology Behavior (Production Defaults)

Default topology mode is `max_model_fast`.

In `max_model_fast`:
- `solo_22gb` runs world=3 on the max-model lane
- `nvlink_pair` is a logical alias lane to that same max-model endpoint
- `solo_3080` is optional dedicated low-latency lane (or aliased when disabled)

In `hetero_3lane`:
- `solo_22gb`: world=1
- `nvlink_pair`: world=2
- `solo_3080`: world=1 (optional, profile-dependent)

Dynamic mapping is inferred from live hardware each start:
- NVLink topology from `nvidia-smi topo -m`
- VRAM tiers from `nvidia-smi --query-gpu=index,memory.total`
- strict validation default: `STRICT_GPU_TOPOLOGY=1`

No-NVLink fallback:
- default behavior allows PCIe pair fallback (`ALLOW_PCIE_PAIR_FALLBACK=1`)
- force hard NVLink-only startup with `REQUIRE_NVLINK=1`

## Generation API And Modes

Single endpoint:
- `POST /v1/generate`

Request fields:
- `prompt` (required)
- `mode` (optional): `retrieval` | `mixed` | `deepseek_only`
- `show_sources` (optional bool)
- `route_hint` (optional): `fast` | `balanced` | `quality`
- `max_new_tokens` (optional)

Default mode is `mixed`.

Mode behavior:
- `retrieval`
  - HRM active
  - sources explicitly injected
  - retrieval-style/citation behavior expected
- `mixed`
  - HRM active
  - silent system prompt active
  - internal source injection (no visible RAG wording expected)
- `deepseek_only`
  - HRM disabled
  - no source injection
  - pure user-prompt path

Response fields of interest:
- `text`
- `mode`
- `source_count`
- `hrm_active`
- `route` (selected backend and routing metadata)
- `sources` only when enabled and mode allows it

Quick calls:

```bash
# mixed (default behavior, silent HRM)
curl -s http://127.0.0.1:8090/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Kurz antworten.","mode":"mixed"}'

# retrieval (explicit source behavior)
curl -s http://127.0.0.1:8090/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Belege nennen.","mode":"retrieval","show_sources":true}'

# deepseek_only (no HRM)
curl -s http://127.0.0.1:8090/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Nur Modellwissen.","mode":"deepseek_only"}'
```

## Source Visibility And Anti-Blackbox Verification

The critical mode logic is in visible source files:
- `hrm_flash/prompt_builder.py`
- `hrm_flash/serve.py`
- `hrm_flash/router.py`

Direct verification command:

```bash
bash scripts/verify_router_source.sh
```

What it verifies:
- runtime modules are loaded from this checkout
- source files compile (`py_compile` + `compileall`)
- mixed silent prompt sentence exists
- `deepseek_only` router log marker exists (`HRM disabled (no retrieval)`)

Startup visibility:
- `scripts/start_native_stack.sh` prints the exact router source path in use
- if `HRM_FLASH_BIN` is set, it prints an explicit external-binary warning

## Build, Start, Stop (Manual Flow)

Bootstrap once (optional but recommended on fresh host):

```bash
PROFILE=deepseek_int8 bash scripts/bootstrap.sh
```

Preflight (default production expectation is 3 GPUs):

```bash
bash scripts/prod_preflight.sh ./model_index 3
```

Start stack:

```bash
bash scripts/start_native_stack.sh ./model_index auto
```

Health smoke:

```bash
bash scripts/smoke_router.sh http://127.0.0.1:8090
```

Stop stack:

```bash
bash scripts/stop_native_stack.sh
```

## Comparable Benchmarks (Real Router Path)

Benchmark script:

```bash
BENCH_MODES=mixed,retrieval,deepseek_only \
BENCH_ROUTE_HINTS=balanced,fast,quality \
BENCH_SCENARIOS=short,medium,long \
BENCH_REPEATS=5 BENCH_WARMUP=1 \
bash scripts/benchmark_deepseek.sh http://127.0.0.1:8090 ./.run/benchmarks/deepseek_live
```

Outputs:
- `raw.csv`
- `summary.csv`
- `summary.txt`

For reproducibility across runs, keep these fixed:
- same model quant (`q8` or `q4`)
- same hardware profile (`A|B|C|D`)
- same route hints/scenarios/repeats
- same running stack revision

## Test And Validation Matrix

Fast mode logic checks:

```bash
python -m unittest tests.test_modes tests.test_serve_modes tests.test_router_logic tests.test_router_source_transparency
```

Full repo test script:

```bash
bash scripts/test.sh
```

Hard script checks (preflight + e2e + benchmark harness):

```bash
RUN_HARD_SCRIPT_TESTS=1 bash scripts/test.sh
```

Dedicated hard scripts:

```bash
bash scripts/test_prod_preflight.sh
bash scripts/test_prod_live_e2e.sh
bash scripts/test_benchmark_deepseek.sh
```

## CI Coverage (GitHub Actions)

Workflow: `.github/workflows/ci-test.yml`

Jobs:
- `test`
  - runs `bash scripts/test.sh` on Ubuntu
- `fresh-install`
  - fresh bootstrap on Ubuntu
  - `pip check` + import checks
  - CLI entrypoint checks
  - regression smoke checks
  - hard production script checks

## Troubleshooting

`ERR: BACKEND=... not supported`
- only `deepseek_int8` is supported in production mainline

`unsupported model ...`
- use one of:
  - `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
  - `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`

`router did not become healthy`
- inspect logs in `./.run/services/`
- especially `router.log`, `solo_22gb.log`, `nvlink_pair.log`, `solo_3080.log`

`no NVLink pair detected`
- default fallback uses PCIe pair
- if hard requirement needed, set `REQUIRE_NVLINK=1`

`preflight failed on GPU count`
- set expected count explicitly:
  - `bash scripts/prod_preflight.sh ./model_index <count>`

## Key Scripts

- `scripts/bootstrap.sh`
- `scripts/prod_preflight.sh`
- `scripts/start_easy.sh`
- `scripts/start_native_stack.sh`
- `scripts/stop_native_stack.sh`
- `scripts/prod_live_e2e.sh`
- `scripts/benchmark_deepseek.sh`
- `scripts/verify_router_source.sh`
- `scripts/test.sh`

## Additional Docs

- `docs/DEEPSEEK_NATIVE_NO_TORCH.md`
- `docs/HETERO_4GPU_STRATEGY.md`
- `docs/NATIVE_MODELS.md`
- `docs/INTEGRATION.md`
- `STACK.md`
