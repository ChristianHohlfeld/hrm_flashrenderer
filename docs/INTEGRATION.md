# Integration Guide (Production, Native DeepSeek)

This repository is production-focused on:
- deterministic HRM retrieval (C++ core)
- native DeepSeek INT8 inference (`deepseek_int8`)
- heterogeneous 4-GPU routing (`22GB`, `11+11 NVLink`, `10GB`)

## Fastest Path

From repo root:

```bash
bash scripts/prod_live_e2e.sh ./model_index "Bitte antworte kurz mit Quellen."
```

This validates build, dependencies, GPU topology, backend health, and a final end-to-end prompt.

## Service Endpoints

After startup (`scripts/start_native_stack.sh`):
- `solo_22gb`: `http://127.0.0.1:8081/v1/health`
- `nvlink_pair`: `http://127.0.0.1:8082/v1/health`
- `solo_3080`: `http://127.0.0.1:8083/v1/health`
- router: `http://127.0.0.1:8090/v1/health`

Inference entrypoint:

```bash
curl -s http://127.0.0.1:8090/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Fasse die relevanten Quellen in 3 Punkten zusammen.","route_hint":"balanced","max_new_tokens":256}'
```

## Routing Controls

Optional request fields for `/v1/generate`:
- `route_hint`: `fast`, `balanced`, `quality`
- `prefer_backend`: `solo_3080`, `nvlink_pair`, `solo_22gb`
- `allow_failover`: `true` or `false`

## Backend Policy

Default backend in CLI and service is `deepseek_int8`.

`torch_tp` remains available as optional fallback profile, but it is not the default production path.

## Systemd Template

Template unit file:
- `scripts/systemd/hrm-flash.service`

It starts/stops the full native stack (`start_native_stack.sh` / `stop_native_stack.sh`) as a `Type=oneshot` service.
