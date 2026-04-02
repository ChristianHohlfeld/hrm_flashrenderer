# HRM FlashRenderer

Native HRM retrieval + native DeepSeek inference for heterogeneous multi-GPU hosts.

## First-User Path (One Command, Full Stack, Final Prompt)

If your server has:
- Python 3.10-3.12 (hard requirement)
- `cmake` + `ctest` (needed when `RUN_BOOTSTRAP=1`)
- NVIDIA driver + CUDA (`nvidia-smi`, `nvcc`)
- first-user default preset `A`: `22/11/11` (3x RTX 2080 Ti, no 3080 required)
- an HRM index at `./model_index` (`router_index.bin` + `index.sqlite`)

run exactly this from repo root:

```bash
bash scripts/prod_live_e2e.sh ./model_index "Bitte antworte auf Deutsch in 3 kurzen Bulletpoints: 1) Stack-Status 2) Kernaussage 3) Route-Hinweis."
```

Or use hardware preset launcher (recommended):

```bash
# A|B|C|D + q8|q4
bash scripts/start_easy.sh ./model_index A q8
```

Then prompt through router:

```bash
curl -s http://127.0.0.1:8090/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Bitte antworte auf Deutsch in 3 kurzen Bulletpoints: 1) Stack-Status 2) Kernaussage 3) Route-Hinweis.","mode":"mixed","route_hint":"balanced","max_new_tokens":256}'
```

This single command does:
1. bootstrap (build + dependencies)
2. production preflight
3. start native DeepSeek stack
4. verify health on all services
5. run strict mode checks against router (`/v1/generate`) for `mixed`, `retrieval`, `deepseek_only`
6. repeat each mode 3x with the same prompt and fail on non-deterministic output
7. fail hard if mode behavior violates requirements (silent leakage, missing retrieval references, non-zero sources in deepseek_only)

If it ends with `E2E PASS`, your pipeline is live.

## If You Do Not Have `./model_index` Yet

Create one (example):

```bash
cat > input.txt <<'EOF'
HRM FlashRenderer verbindet deterministische Retrieval-Quellen mit nativer DeepSeek-Inferenz.
Die Topologie nutzt drei Dienste: solo_22gb, nvlink_pair, solo_3080.
EOF
bash scripts/build.sh
bash scripts/make_model.sh input.txt ./model_index 200
```

## If Bootstrap Already Done

Skip reinstall/rebuild:

```bash
RUN_BOOTSTRAP=0 bash scripts/prod_live_e2e.sh ./model_index "Bitte antworte auf Deutsch in 3 kurzen Bulletpoints: 1) Stack-Status 2) Kernaussage 3) Route-Hinweis."
```

With fixed preset during E2E (recommended):

```bash
RUN_BOOTSTRAP=0 EXPECTED_GPUS=3 HW_BASE_PROFILE=A MODEL_QUANT=q8 ENABLE_SOLO_3080=0 \
  bash scripts/prod_live_e2e.sh ./model_index "Bitte antworte kurz."
```

## Keep Running / Stop

By default, stack stays up after E2E PASS.

Stop:

```bash
bash scripts/stop_native_stack.sh
```

Auto-stop right after E2E test:

```bash
AUTO_STOP=1 bash scripts/prod_live_e2e.sh ./model_index "Bitte antworte kurz: Stack OK?"
```

## Manual Flow (If You Prefer Explicit Steps)

```bash
PROFILE=deepseek_int8 bash scripts/bootstrap.sh
bash scripts/prod_preflight.sh ./model_index 4
bash scripts/start_native_stack.sh ./model_index auto
bash scripts/smoke_router.sh http://127.0.0.1:8090
```

Final prompt call:

```bash
curl -s http://127.0.0.1:8090/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Bitte antworte auf Deutsch in 3 kurzen Bulletpoints: 1) Stack-Status 2) Kernaussage 3) Route-Hinweis.","mode":"mixed","route_hint":"balanced","max_new_tokens":256}'
```

## Model + Hardware Alignment (Default)

Default startup mode is `TOPOLOGY_MODE=max_model_fast` with `auto` profile (`max_vram_hetero`):
- max-model lane (`solo_22gb`, world=3 on `22GB + 11GB + 11GB NVLink pair`):
  `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
- optional second lane (`solo_3080`, world=1 on `10GB` normal RTX 3080) can be enabled per preset/env
- router `balanced` and `quality` target the max-model lane (nvlink logical lane is an alias)
- router `fast` uses 3080 lane only when enabled; otherwise aliases to max-model lane

Per-service defaults in `max_model_fast`:
- 32B lane: `max_seq_len=3072`, `prefill_chunk_size=512`
- optional 3080 lane: `max_seq_len=3072`, `prefill_chunk_size=384`

Supported DeepSeek models in production mainline (fixed allowlist):
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
- `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` (use `MODEL_QUANT=q4`)

Hardware presets (easy launcher `scripts/start_easy.sh`):
- `A`: 22GB + 11GB + 11GB, no NVLink required, 3080 lane disabled
- `B`: 22GB + 11GB + 11GB + 10GB (RTX 3080), no NVLink required
- `C`: 22GB + 11GB + 11GB + 10GB (RTX 3080), NVLink required on 11GB pair
- `D`: 22GB + 22GB + 11GB + 11GB, 3080 lane disabled

Direct equivalent (without helper):

```bash
HW_BASE_PROFILE=A MODEL_QUANT=q8 ENABLE_SOLO_3080=0 bash scripts/start_native_stack.sh ./model_index auto
```

## Generation Modes (Single Endpoint)

`POST /v1/generate` supports three modes via JSON field `mode`:
- `mixed` (default): HRM runs deterministically in background (`top_k=16`, `k=16`, `1.8s`), sources are internal/hidden, and the silent system prompt explicitly forbids source/retrieval mentions.
- `retrieval`: HRM sources are explicit in prompt context and model is instructed to cite source ids.
- `deepseek_only`: no HRM query, no source injection, pure model response path (raw user prompt, no system/retrieval context).

Default behavior is `mixed`, and silent mode is enforced: normal answers must not contain phrases like `laut den Quellen`, `aus den Snippets`, or `basierend auf ...`.

Router default mode can be pinned at startup:

```bash
ROUTER_DEFAULT_MODE=mixed bash scripts/start_native_stack.sh ./model_index auto
```

Optional JSON flag:
- `show_sources`: include `sources` array in HTTP response (off by default, ignored for `deepseek_only`).
- default behavior: `retrieval` auto-enables source output; `mixed` keeps sources hidden unless explicitly requested.
- `hrm_active` in response: `true` for `mixed`/`retrieval`, `false` for `deepseek_only`.

`scripts/prod_live_e2e.sh` performs strict mode verification by default:
- `RUN_MODE_MATRIX=1` (default): tests all three modes with different prompts.
- `E2E_DETERMINISM_RUNS=3` (default): repeats each mode 3x and fails on drift.
- `RUN_MODE_MATRIX=0`: test a single mode selected by `ROUTER_MODE`.

Example:

```json
{
  "prompt": "Dein normaler Prompt",
  "mode": "mixed",
  "show_sources": false
}
```

Mode quick checks:

```bash
# mixed (silent HRM, hidden sources)
curl -s http://127.0.0.1:8090/v1/generate -H 'Content-Type: application/json' -d '{"prompt":"Kurz antworten.","mode":"mixed"}'

# retrieval (explicit sources/citations behavior)
curl -s http://127.0.0.1:8090/v1/generate -H 'Content-Type: application/json' -d '{"prompt":"Belege nennen.","mode":"retrieval"}'

# deepseek_only (no HRM)
curl -s http://127.0.0.1:8090/v1/generate -H 'Content-Type: application/json' -d '{"prompt":"Nur Modellwissen.","mode":"deepseek_only"}'
```

Single-mode E2E run (optional):

```bash
RUN_MODE_MATRIX=0 ROUTER_MODE=mixed bash scripts/prod_live_e2e.sh ./model_index "Bitte antworte kurz."
```

### Mode-Implementation Proof (Source, not binary-only)

The mode behavior is implemented in visible Python source files (not only in a compiled artifact):
- `hrm_flash/prompt_builder.py`
  - `MIXED_SYSTEM_PROMPT` (silent prompt)
  - `build_mixed_prompt`, `build_retrieval_prompt`, `build_deepseek_only_prompt`
- `hrm_flash/serve.py`
  - `_build_prompt`: `deepseek_only` returns early (no `run_hrm_query` call)
  - `/v1/generate`: `hrm_active` and source exposure are mode-dependent
  - runtime `audit` fields per response: `hrm_called`, `source_injected_count`, `prompt_template`
  - `/v1/health` audit counters: `hrm_query_calls_total`, `hrm_query_calls_by_mode`, `mode_counts`
- `hrm_flash/router.py`
  - `_resolve_mode`, `_resolve_show_sources`
  - `deepseek_only` forces `show_sources=False` and logs `mode=deepseek_only HRM disabled (no retrieval)`
- `scripts/start_native_stack.sh`
  - validates `ROUTER_DEFAULT_MODE` (`retrieval|mixed|deepseek_only`)
  - passes `--default_mode` into router startup

Quick verification commands:

```bash
# Unit-level mode logic
python -m unittest tests.test_modes tests.test_serve_modes tests.test_router_logic

# Hard script checks (includes mode-matrix + determinism)
RUN_HARD_SCRIPT_TESTS=1 bash scripts/test.sh

# End-to-end mode matrix only
bash scripts/test_prod_live_e2e.sh
```

## Comparable DeepSeek Benchmarks (Real Router Path)

Use the dedicated benchmark runner for reproducible latency/throughput comparisons:

```bash
BENCH_MODES=mixed,deepseek_only,retrieval \
BENCH_ROUTE_HINTS=balanced,fast,quality \
BENCH_SCENARIOS=short,medium,long \
BENCH_REPEATS=5 BENCH_WARMUP=1 \
bash scripts/benchmark_deepseek.sh http://127.0.0.1:8090 ./.run/benchmarks/deepseek_live
```

Output files:
- `raw.csv` (all runs)
- `summary.csv` (p50/p95/mean by scenario+mode+route_hint)
- `summary.txt` (top combinations by score = mean_tokens_per_s / p95_latency_ms)

The start scripts now pin runtime to this checkout by default (`python -m hrm_flash.cli`) to avoid stale global `hrm-flash` binaries.
Optional override for advanced setups:

```bash
HRM_FLASH_BIN=/usr/local/bin/hrm-flash bash scripts/start_native_stack.sh ./model_index auto
```

## Dynamic GPU Mapping (No Static Order Assumptions)

`scripts/start_native_topology.sh` always infers:
- NVLink pair from `nvidia-smi topo -m`
- VRAM tiers from `nvidia-smi --query-gpu=index,memory.total`
- optional fixed hardware profile via `HW_BASE_PROFILE=auto|A|B|C|D`

Strict mode is on by default:
- `STRICT_GPU_TOPOLOGY=1`
- startup fails on mapping mismatch

If NVLink bridge is temporarily missing, startup still works by default:
- falls back to best PCIe pair (`ALLOW_PCIE_PAIR_FALLBACK=1`, default)
- prints a warning and continues

To enforce hard NVLink-only startup:
- `REQUIRE_NVLINK=1`

## Backend Scope

Main production backend:
- `deepseek_int8` (default)

Native DeepSeek path supports dense HF safetensors layouts and does not support MoE/GGUF/GPTQ/AWQ in this backend.

## Key Scripts

- `scripts/prod_live_e2e.sh`
- `scripts/prod_preflight.sh`
- `scripts/benchmark_deepseek.sh`
- `scripts/start_easy.sh`
- `scripts/start_native_stack.sh`
- `scripts/stop_native_stack.sh`
- `scripts/smoke_router.sh`
- `scripts/test.sh`
- `scripts/test_topology_detection.sh`

## Additional Docs

- `docs/DEEPSEEK_NATIVE_NO_TORCH.md`
- `docs/HETERO_4GPU_STRATEGY.md`
- `docs/NATIVE_MODELS.md`
- `docs/INTEGRATION.md`
- `STACK.md`
