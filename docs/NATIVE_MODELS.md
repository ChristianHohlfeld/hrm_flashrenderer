# Native Model Compatibility (hrm-flash)

This document describes what the native `hrm-flash` path can run today.

## Goal

Run regular HF safetensors models directly through `hrm-flash` (native TP path),
not only through the GGUF `renderer/hrm_render.py` wrapper.

## Backend

- `deepseek_int8` (native no-torch DeepSeek INT8 engine path)

Production mainline is deepseek-only.

## Supported now

- Dense decoder-only model layouts that expose Llama/Qwen-style keys:
  - `model.embed_tokens.weight`
  - `model.layers.*.self_attn.{q,k,v,o}_proj.weight`
  - `model.layers.*.mlp.{gate,up,down}_proj.weight`
  - `model.norm.weight`
- Head dimension must be `64` or `128` (kernel constraint).
- TP world validation is enforced for hidden/heads/intermediate/vocab divisibility.

## Not supported in native path

- Full MoE model families (for example DeepSeek-V3 style MoE configs).
- GPTQ/AWQ-style quantized checkpoint layouts for this loader.
- GGUF files (those belong to `renderer/hrm_render.py`).

## Practical DeepSeek guidance

- `deepseek_int8`: uses exported `model_q8.bin` + native engine process, with
  prompt token prefill and decoded output (no torch runtime in this path).
- For heterogeneous `22/11/11/10 GB` production hosts, default max-size target is:
  `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` on world=3 (`22GB + NVLink 11+11`).

## Example (native generate)

```bash
hrm-flash generate \
  --hrm_model ./model \
  --llm_model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --backend deepseek_int8 \
  --world 3 \
  --prompt "Explain the retrieval evidence briefly."
```

## Example (native HTTP service)

```bash
hrm-flash serve \
  --hrm_model ./model \
  --llm_model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --backend deepseek_int8 \
  --world 3 \
  --host 0.0.0.0 \
  --port 8081
```

`generate` and `serve` perform an explicit native compatibility preflight and
fail fast with a clear message if the selected model layout is unsupported.
