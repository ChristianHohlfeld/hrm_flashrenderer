# Stack Overview

## Production Path

Mainline production stack:
1. HRM retrieval (`hrm_core`)
2. Native DeepSeek INT8 inference (`deepseek_int8`)
3. Heterogeneous router across:
   - `solo_22gb`
   - `nvlink_pair`
   - `solo_3080`

Primary startup flow:
- `scripts/prod_live_e2e.sh`

## Backend Policy

Production mainline is deepseek-only:
- `deepseek_int8`

Non-native torch fallback is intentionally blocked in production scripts and CLI backend options.

## Hardware Strategy

For `22/11/11/10 GB` with a single NVLink bridge on the two 11GB cards:
- do not use `world=4` as default
- default mode `max_model_fast` runs:
  - one world=3 max-model lane (`22GB + 11GB + 11GB`) with DeepSeek 32B
  - one world=1 fast lane (`10GB`) with DeepSeek 7B
  - router keeps three logical lanes; `balanced` + `quality` map to the 32B lane
- optional mode `hetero_3lane` runs separate `22GB`, `11+11 NVLink`, `10GB` services
- route per request class (`fast`, `balanced`, `quality`)

## Dependency Profiles

- `requirements.prod.txt`: lean no-torch core
- `requirements.server.txt`: HTTP server

## Key Guarantees

- strict NVLink pair detection via `nvidia-smi topo -m`
- no static GPU order assumptions
- fail-fast preflight for missing binaries, model index, and incompatible runtime
