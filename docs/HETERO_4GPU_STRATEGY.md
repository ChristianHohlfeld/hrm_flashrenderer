# Heterogeneous 4-GPU Native Strategy (22/11/11/10 GB)

Target host:
- RTX 2080 Ti 22 GB (solo)
- RTX 2080 Ti 11 GB + RTX 2080 Ti 11 GB (NVLink pair)
- RTX 3080 Ti 10 GB (solo)
- PCIe-era board, older server platform

## Why not `world=4` default

In this topology, tensor-parallel across all 4 GPUs is usually slower and less stable:
- TP step latency is gated by the slowest shard and weakest interconnect.
- You do not have full-mesh NVLink; only one 11+11 pair is linked.
- PCIe traffic to the unlinked cards adds synchronization overhead.

Use multiple native services instead of one global TP graph.

## Recommended service topology

Default production mode (`TOPOLOGY_MODE=max_model_fast`) prioritizes model size:
- max-model lane (`solo_22gb`): `world=3` on `22GB + 11GB + 11GB NVLink pair`
- fast lane (`solo_3080`): `world=1` on `10GB`
- `nvlink_pair` remains a logical router lane alias to the same max-model endpoint

Optional compatibility mode (`TOPOLOGY_MODE=hetero_3lane`):
- `solo_22gb` (`world=1`)
- `nvlink_pair` (`world=2`)
- `solo_3080` (`world=1`)

This repo already includes helper scripts:
- `scripts/start_native_topology.sh`
- `scripts/stop_native_topology.sh`
- `scripts/start_native_stack.sh`
- `scripts/stop_native_stack.sh`

You can set per-service model overrides in `start_native_topology.sh`:
- `LLM_MODEL_SOLO_22GB`
- `LLM_MODEL_NVLINK_PAIR`
- `LLM_MODEL_SOLO_3080`

Default `auto` profile (`MODEL_PROFILE=max_vram_hetero`, `TOPOLOGY_MODE=max_model_fast`):
- max-model lane: `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
- fast lane: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`

GPU mapping is dynamically inferred (no static ordering assumptions):
- Detect NVLink pair from `nvidia-smi topo -m`
- Detect VRAM tiers from `nvidia-smi --query-gpu=index,memory.total`
- Enforce strict mapping validation by default (`STRICT_GPU_TOPOLOGY=1`)
- If NVLink is temporarily unavailable, default behavior is PCIe pair fallback (`ALLOW_PCIE_PAIR_FALLBACK=1`)
- Set `REQUIRE_NVLINK=1` if startup must fail without an NVLink pair

## One-command startup

Start full topology plus router:

```bash
bash scripts/start_native_stack.sh ./model_index auto
```

Optional legacy split:

```bash
TOPOLOGY_MODE=hetero_3lane bash scripts/start_native_stack.sh ./model_index auto
```

Build/export native DeepSeek assets explicitly (optional, done automatically by
`start_native_stack.sh` when `BACKEND=deepseek_int8`):

```bash
bash scripts/build_deepseek_native.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
bash scripts/build_deepseek_native.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

Engine binary default output:
- `./.run/bin/deepseek_engine`
Default native build arch list:
- `CUDA_ARCH_LIST=75,86`

Native engine layer placement is balanced across visible GPUs, so the 11+11 NVLink pair is no longer front-heavy on one card.

Router endpoint for clients:

```bash
curl -s http://127.0.0.1:8090/v1/health
curl -s http://127.0.0.1:8090/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Summarize the top retrieval evidence in 5 bullets."}'
```

Optional router request fields:
- `route_hint`: `fast`, `balanced`, or `quality`
- `prefer_backend`: `solo_3080`, `nvlink_pair`, `solo_22gb`
- `allow_failover`: `true` or `false`

Quick smoke test:

```bash
bash scripts/smoke_router.sh http://127.0.0.1:8090
```

Startup reliability knobs:
- `STARTUP_WAIT_TIMEOUT_S` (default `240`)
- `STARTUP_POLL_INTERVAL_S` (default `2`)

## Dependency profiles

- `requirements.prod.txt`: lean no-torch core.
- `requirements.server.txt`: HTTP server dependencies.

## Model policy for native path

Native loader supports dense safetensors Llama/Qwen-style layouts.
It does not support full MoE (DeepSeek-V3 style), GPTQ/AWQ layouts, or GGUF.

Use:
- DeepSeek distill dense safetensors variants (for example Llama-based distills)
- Other dense HF safetensors decoder models with compatible keys

Avoid:
- Full DeepSeek MoE checkpoints for native path
- GGUF in native path (use renderer fallback only when needed)

## Routing policy (gateway layer)

Use a thin router (reverse proxy or app logic) in front of the 3 endpoints:
- In default `max_model_fast`, `balanced` and `quality` land on the same 32B endpoint.
- Send explicit low-latency prompts to `fast` (`solo_3080`).
- Keep failover enabled for operational resilience.

Start conservative and adapt from measurements:
- `max_concurrent=1` per service
- queue at gateway level
- route by prompt token count and SLA class

## Suggested starting params

- max-model lane (`world=3`): `max_seq_len=3072`, `prefill_chunk_size=512`
- `solo_3080`: `max_seq_len=3072`, `prefill_chunk_size=384`

Tune in this order:
1. Keep tail latency stable under load.
2. Increase context only after OOM-free runs.
3. Increase chunk size only if prefill is bottlenecked.

## Build/runtime notes for mixed 2080/3080 hosts

- Build kernels for both arch families:
  - `CUDA_ARCH_LIST=75,86`
- Keep separate service processes pinned by `CUDA_VISIBLE_DEVICES`.
- Prefer local model cache (`llm_models/`) to avoid startup variance.
- Validate model compatibility before production rollout (native preflight is now built in).

## Rollout sequence

1. Start default `max_model_fast` topology and verify the 32B lane is healthy.
2. Run health checks and short canary prompts on each endpoint.
3. Run mixed load for 30-60 minutes, monitor OOM and p95 latency.
4. Freeze routing thresholds and only then raise context windows.
