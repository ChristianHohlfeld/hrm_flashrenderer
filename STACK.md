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

## Optional Path

Optional fallback profile:
- `torch_tp` (requires `requirements.torch.txt`)

This path is not the default.

## Hardware Strategy

For `22/11/11/10 GB` with a single NVLink bridge on the two 11GB cards:
- do not use `world=4` as default
- run three services with dynamic GPU topology detection
- route per request class (`fast`, `balanced`, `quality`)

## Dependency Profiles

- `requirements.prod.txt`: lean no-torch core
- `requirements.server.txt`: HTTP server
- `requirements.torch.txt`: optional torch profile

## Key Guarantees

- strict NVLink pair detection via `nvidia-smi topo -m`
- no static GPU order assumptions
- fail-fast preflight for missing binaries, model index, and incompatible runtime
