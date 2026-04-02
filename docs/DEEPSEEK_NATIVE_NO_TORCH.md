# DeepSeek Native No-Torch Runbook

This runbook uses the `deepseek_int8` backend in the main `hrm-flash` flow.
Recommended runtime: Python 3.10-3.12 on Linux hosts.

## 1) Install minimal profiles

```bash
python3 -m pip install -r requirements.prod.txt
python3 -m pip install -r requirements.server.txt
python3 -m pip install -e .
```

Production mainline is deepseek-only and does not require `requirements.torch.txt`.

## 2) Build/export DeepSeek native assets

```bash
bash scripts/build_deepseek_native.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
bash scripts/build_deepseek_native.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

For mixed 2080/3080 hosts, default build is fatbin for `75,86`.
Override if needed:

```bash
CUDA_ARCH_LIST=75,86 bash scripts/build_deepseek_native.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
```

This prepares:
- `./llm_models/<model>/model_q8.bin`
- `./.run/bin/deepseek_engine` (native CUDA binary)

## 3) Start full hetero stack

```bash
export BACKEND=deepseek_int8
export PREPARE_MODELS=1
bash scripts/start_native_stack.sh ./model_index auto
```

Default `auto` profile with `TOPOLOGY_MODE=max_model_fast`:
- max-model lane (`solo_22gb`, world=3 on `22GB + NVLink 11+11`):
  `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
- fast lane (`solo_3080`, world=1 on `10GB`):
  `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- `nvlink_pair` is a logical alias lane to the same 32B endpoint (for router compatibility)

If you need the previous explicit 3-lane split:

```bash
TOPOLOGY_MODE=hetero_3lane bash scripts/start_native_stack.sh ./model_index auto
```

GPU topology is dynamically inferred with strict NVLink validation by default (`STRICT_GPU_TOPOLOGY=1`).

## 4) Query router

```bash
curl -s http://127.0.0.1:8090/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain the top retrieval evidence briefly.","route_hint":"balanced"}'
```

## 5) Stop stack

```bash
bash scripts/stop_native_stack.sh
```
