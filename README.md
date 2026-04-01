# HRM FlashRenderer

Native HRM retrieval plus native DeepSeek INT8 inference for heterogeneous multi-GPU hosts.

## Production Goal

This repository is finalized around one production flow:

1. Native DeepSeek INT8 backend (`deepseek_int8`) as the default path.
2. Heterogeneous 4-GPU topology with router-based dispatch (no global `world=4` bottleneck).
3. Strict dynamic GPU topology inference with explicit NVLink pair detection.

Target host profile:
- RTX 2080 Ti 22 GB (solo)
- RTX 2080 Ti 11 GB + RTX 2080 Ti 11 GB (NVLink pair)
- RTX 3080 Ti 10 GB (solo)

## Hardware-Aligned Model Profile (Max VRAM Without OOM)

Default startup mode is `auto` with profile `max_vram_hetero`:

- `solo_22gb`: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
- `nvlink_pair`: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
- `solo_3080`: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`

Default per-service runtime settings:

- `solo_22gb`: `max_seq_len=4096`, `prefill_chunk_size=768`
- `nvlink_pair`: `max_seq_len=4096`, `prefill_chunk_size=768`
- `solo_3080`: `max_seq_len=3072`, `prefill_chunk_size=384`

## Dynamic GPU Topology (Always Inferred)

`scripts/start_native_topology.sh` now enforces dynamic detection:

- Detects GPU VRAM via `nvidia-smi --query-gpu=index,memory.total`.
- Detects NVLink connectivity via `nvidia-smi topo -m`.
- Selects NVLink service pair from actual NVLink links.
- Selects `solo_22gb` as largest remaining VRAM GPU.
- Selects `solo_3080` as smallest remaining VRAM GPU.

Strict behavior (`STRICT_GPU_TOPOLOGY=1`, default):

- Startup fails if topology cannot be inferred.
- Startup fails if manual `GPU_*` overrides mismatch inferred topology.
- No static fallback ordering is used by default.

## Dependencies

Lean production runtime:

```bash
python3 -m pip install -r requirements.prod.txt
python3 -m pip install -r requirements.server.txt
python3 -m pip install -e .
```

Optional torch backend only:

```bash
python3 -m pip install -r requirements.torch.txt
```

Native model path uses dense HF safetensors layouts and does not support MoE/GGUF/GPTQ/AWQ for this backend.

## End-to-End Production Flow (Server)

### 1) Preflight

```bash
bash scripts/prod_preflight.sh ./model_index 4
```

Preflight now includes:
- script syntax validation
- Python import checks
- CLI entrypoint checks
- topology detection self-test (`scripts/test_topology_detection.sh`)

### 2) Optional explicit model exports

```bash
bash scripts/build_deepseek_native.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
bash scripts/build_deepseek_native.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

### 3) Start full native stack

```bash
export BACKEND=deepseek_int8
export PREPARE_MODELS=1
bash scripts/start_native_stack.sh ./model_index auto
```

### 4) Verify health

```bash
curl -s http://127.0.0.1:8081/v1/health
curl -s http://127.0.0.1:8082/v1/health
curl -s http://127.0.0.1:8083/v1/health
curl -s http://127.0.0.1:8090/v1/health
```

### 5) Router smoke test

```bash
bash scripts/smoke_router.sh http://127.0.0.1:8090
```

### 6) Query inference

```bash
curl -s http://127.0.0.1:8090/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Summarize top retrieval evidence in 5 bullets.","route_hint":"balanced"}'
```

### 7) Stop stack

```bash
bash scripts/stop_native_stack.sh
```

## Validation Status (Before vs After)

### Before finalization (issues)

- Static GPU mapping assumptions (`1,2` / `0` / `3`) in topology startup.
- Main docs used small default model examples.
- Native engine had fixed layer split (`10/9/rest`) that could overload one GPU in pair mode.
- `scripts/test.sh` was brittle for mixed environments (hard fail when `cmake` or `python` binary naming differed).
- CRLF shell script issues could break Bash execution (`pipefail\r`).

### After finalization (implemented)

- Auto profile defaults to `14B/14B/7B` for this host shape.
- Strict dynamic topology inference with NVLink pair detection is enforced.
- Engine layer placement changed to balanced per visible GPU count.
- Startup scripts support `auto` model profile and per-service tuning defaults.
- Added topology detection regression test:
  - `scripts/test_topology_detection.sh`
- Hardened `scripts/test.sh` to run what is available and skip unsupported local checks cleanly.

### Executed checks in this workspace

- `bash -n scripts/*.sh` on production scripts: PASS
- `python -m compileall hrm_flash/serve.py hrm_flash/deepseek_native.py hrm_flash/cli.py hrm_flash/router.py`: PASS
- `python -m hrm_flash.cli --help`, `serve --help`, `router --help`, `generate --help`: PASS
- `bash scripts/test_topology_detection.sh`: PASS
- `bash scripts/test.sh`: PASS (with environment-driven skips for unavailable tools)
- `bash scripts/prod_preflight.sh ./.run/verify_model 1`: FAIL in this desktop environment (`nvidia-smi` missing)

That final preflight failure is expected on this non-GPU dev environment and must be re-run on your production server.

## Required Server Acceptance (Must Pass Before Prod Switch)

1. `bash scripts/prod_preflight.sh ./model_index 4`
2. `bash scripts/start_native_stack.sh ./model_index auto`
3. `bash scripts/smoke_router.sh http://127.0.0.1:8090`
4. 10-20 real prompts through router with `route_hint` values `fast`, `balanced`, `quality`
5. Confirm no OOM / no service restarts in `.run/services/*.log`
6. `bash scripts/stop_native_stack.sh` and clean restart once

## Key Scripts

- `scripts/start_native_stack.sh`
- `scripts/start_native_topology.sh`
- `scripts/build_deepseek_native.sh`
- `scripts/deepseek_native_engine.sh`
- `scripts/prod_preflight.sh`
- `scripts/smoke_router.sh`
- `scripts/test.sh`
- `scripts/test_topology_detection.sh`

## Additional Docs

- `docs/DEEPSEEK_NATIVE_NO_TORCH.md`
- `docs/HETERO_4GPU_STRATEGY.md`
- `docs/NATIVE_MODELS.md`
