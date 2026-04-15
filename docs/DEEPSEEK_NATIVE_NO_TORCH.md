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

## 2) Select hardware pool (mandatory)

```bash
bash scripts/hw_select.sh
```

All native build/start scripts require this selection file.
Default writes: `11gb=2`, `22gb=1`, `3080_10gb=0`, `require_nvlink_11gb_pair=1`, `quant=q8`.

## 3) Build/export DeepSeek native assets

```bash
bash scripts/build_deepseek_native.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
bash scripts/build_deepseek_native.sh deepseek-ai/DeepSeek-R1-Distill-Llama-70B
```

`CUDA_ARCH_LIST` is derived from selected hardware (`75` or `75,86`).
Override is rejected if it conflicts with the selected pool.

When hardware selection changes, rebuild is forced automatically.

This prepares:
- `./llm_models/<model>/model_q8.bin`
- `./.run/bin/deepseek_engine` (native CUDA binary)

## 4) Start full native stack

```bash
export BACKEND=deepseek_int8
export PREPARE_MODELS=1
bash scripts/start_native_stack.sh ./model_index auto
```

`TOPOLOGY_MODE` is derived from selected hardware:
- `max_model_fast` for full `22 + 11 + 11` lane availability
- `single_lane` for reduced pools

GPU topology is dynamically inferred with strict NVLink validation by default (`STRICT_GPU_TOPOLOGY=1`).

## 5) Query router

```bash
curl -s http://127.0.0.1:8090/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain the top retrieval evidence briefly.","route_hint":"balanced"}'
```

## 6) Stop stack

```bash
bash scripts/stop_native_stack.sh
```
