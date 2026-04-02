# HRM FlashRenderer

Native HRM retrieval + native DeepSeek INT8 inference for heterogeneous multi-GPU hosts.

## First-User Path (One Command, Full Stack, Final Prompt)

If your server has:
- Python 3.10-3.12 (hard requirement)
- `cmake` + `ctest` (needed when `RUN_BOOTSTRAP=1`)
- NVIDIA driver + CUDA (`nvidia-smi`, `nvcc`)
- 4 GPUs (22/11/11/10 GB) with one NVLink pair on the two 11 GB cards
- an HRM index at `./model_index` (`router_index.bin` + `index.sqlite`)

run exactly this from repo root:

```bash
bash scripts/prod_live_e2e.sh ./model_index "Bitte antworte auf Deutsch in 3 kurzen Bulletpoints: 1) Stack-Status 2) Wichtigster Fakt aus den Quellen 3) Route-Hinweis."
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
RUN_BOOTSTRAP=0 bash scripts/prod_live_e2e.sh ./model_index "Bitte antworte auf Deutsch in 3 kurzen Bulletpoints: 1) Stack-Status 2) Wichtigster Fakt aus den Quellen 3) Route-Hinweis."
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
  -d '{"prompt":"Bitte antworte auf Deutsch in 3 kurzen Bulletpoints: 1) Stack-Status 2) Wichtigster Fakt aus den Quellen 3) Route-Hinweis.","route_hint":"balanced","max_new_tokens":256}'
```

## Model + Hardware Alignment (Default)

`auto` startup profile (`max_vram_hetero`) uses:
- `solo_22gb`: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
- `nvlink_pair`: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
- `solo_3080`: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`

Per-service defaults:
- `solo_22gb`: `max_seq_len=4096`, `prefill_chunk_size=768`
- `nvlink_pair`: `max_seq_len=4096`, `prefill_chunk_size=768`
- `solo_3080`: `max_seq_len=3072`, `prefill_chunk_size=384`

## Dynamic GPU Mapping (No Static Order Assumptions)

`scripts/start_native_topology.sh` always infers:
- NVLink pair from `nvidia-smi topo -m`
- VRAM tiers from `nvidia-smi --query-gpu=index,memory.total`

Strict mode is on by default:
- `STRICT_GPU_TOPOLOGY=1`
- startup fails if detected topology is missing/mismatched

## Backend Scope

Main production backend:
- `deepseek_int8` (default)

Optional:
- `torch_tp` (separate dependency profile)

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
