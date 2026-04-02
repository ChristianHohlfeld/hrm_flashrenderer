# Integration Guide (Production, Native DeepSeek)

This repository is production-focused on:
- deterministic HRM retrieval (C++ core)
- native DeepSeek inference (`deepseek_int8`, quantized model bins)
- heterogeneous 4-GPU routing (`22GB`, `11+11 NVLink`, `10GB`)

## Fastest Path

From repo root:

```bash
bash scripts/prod_live_e2e.sh ./model_index "Bitte antworte kurz und präzise."
```

This validates build, dependencies, GPU topology, backend health, and a final end-to-end prompt.

## Service Endpoints

After startup (`scripts/start_native_stack.sh`):
- `solo_22gb`: `http://127.0.0.1:8081/v1/health`
- `nvlink_pair`: `http://127.0.0.1:8081/v1/health` (logical alias lane in default `max_model_fast` mode)
- `solo_3080`: `http://127.0.0.1:8083/v1/health`
- router: `http://127.0.0.1:8090/v1/health`

Default mode is `TOPOLOGY_MODE=max_model_fast`:
- max-model lane: DeepSeek 32B on `22GB + 11GB + 11GB` (world=3)
- fast lane: DeepSeek 7B on `10GB` (world=1)

Optional:
- `TOPOLOGY_MODE=hetero_3lane` for explicit separate `22GB` / `11+11 NVLink` / `10GB` lanes.

Inference entrypoint:

```bash
curl -s http://127.0.0.1:8090/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Fasse die relevanten Quellen in 3 Punkten zusammen.","mode":"mixed","route_hint":"balanced","max_new_tokens":256}'
```

Generation modes (same endpoint):
- `mixed` (default): silent HRM in background
- `retrieval`: explicit source/citation mode
- `deepseek_only`: no HRM

Router startup default for mode:

```bash
ROUTER_DEFAULT_MODE=mixed bash scripts/start_native_stack.sh ./model_index auto
```

## Routing Controls

Optional request fields for `/v1/generate`:
- `route_hint`: `fast`, `balanced`, `quality`
- `prefer_backend`: `solo_3080`, `nvlink_pair`, `solo_22gb`
- `allow_failover`: `true` or `false`

## Backend Policy

Default backend in CLI and service is `deepseek_int8`.
Production mainline is deepseek-only. Non-native torch fallback is intentionally blocked in production scripts and CLI backend options.

## Systemd Template

Template unit file:
- `scripts/systemd/hrm-flash.service`

It starts/stops the full native stack (`start_native_stack.sh` / `stop_native_stack.sh`) as a `Type=oneshot` service.
