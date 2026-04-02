# HRM FlashRenderer

Native HRM retrieval + native DeepSeek inference for heterogeneous multi-GPU hosts.

## First-User Path (One Command, Full Stack, Final Prompt)

If your server has:
- Python 3.10-3.12 (hard requirement)
- `cmake` + `ctest` (needed when `RUN_BOOTSTRAP=1`)
- NVIDIA driver + CUDA (`nvidia-smi`, `nvcc`)
- 4 GPUs (22/11/11/10 GB) with one NVLink pair on the two 11 GB cards
- an HRM index at `./model_index` (`router_index.bin` + `index.sqlite`)

run exactly this from repo root:

```bash
bash scripts/prod_live_e2e.sh ./model_index "Bitte antworte auf Deutsch in 3 kurzen Bulletpoints: 1) Stack-Status 2) Kernaussage 3) Route-Hinweis."
```

This single command does:
1. bootstrap (build + dependencies)
2. production preflight
3. start native DeepSeek stack
4. verify health on all services
5. send final prompt through router (`/v1/generate`)
6. fail hard if no real retrieval+inference happened

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
- fast lane (`solo_3080`, world=1 on `10GB`):
  `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- router `balanced` and `quality` both target the 32B lane (nvlink logical lane is an alias)
- router `fast` targets the 3080 lane

Per-service defaults in `max_model_fast`:
- 32B lane: `max_seq_len=3072`, `prefill_chunk_size=512`
- 3080 lane: `max_seq_len=3072`, `prefill_chunk_size=384`

If you explicitly want the old 3-lane split (14B/14B/7B), use:

```bash
TOPOLOGY_MODE=hetero_3lane bash scripts/start_native_stack.sh ./model_index auto
```

## Generation Modes (Single Endpoint)

`POST /v1/generate` supports three modes via JSON field `mode`:
- `mixed` (default): HRM runs deterministically in background (`top_k=16`, `k=16`, `1.8s`), sources are internal/hidden, model must not mention retrieval.
- `retrieval`: HRM sources are explicit in prompt context and model is instructed to cite source ids.
- `deepseek_only`: no HRM query, no source injection, pure model response path.

Router default mode can be pinned at startup:

```bash
ROUTER_DEFAULT_MODE=mixed bash scripts/start_native_stack.sh ./model_index auto
```

Optional JSON flag:
- `show_sources`: include `sources` array in HTTP response (off by default, ignored for `deepseek_only`).

Example:

```json
{
  "prompt": "Dein normaler Prompt",
  "mode": "mixed",
  "show_sources": false
}
```

## Dynamic GPU Mapping (No Static Order Assumptions)

`scripts/start_native_topology.sh` always infers:
- NVLink pair from `nvidia-smi topo -m`
- VRAM tiers from `nvidia-smi --query-gpu=index,memory.total`

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
